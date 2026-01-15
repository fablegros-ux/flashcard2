import os, re, csv, io, zipfile
import tempfile
from typing import List, Dict, Tuple, Optional
import streamlit as st

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import Frame, Paragraph, KeepInFrame
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader
from PIL import Image

# Install missing packages if running in an environment where they might not persist
print("Installation des dépendances...")
#!pip install -q streamlit reportlab Pillow
print("Dépendances installées.")

# ----------------------------
# Réglages
# ----------------------------
OUTPUT_PDF = "cartes_recto_verso.pdf"
NB_CARTES = 10
COLS, ROWS = 2, 5            # 2 x 5 = 10 cartes
MARGIN = 1.0 * cm
GAP = 0.35 * cm              # espace entre cartes (découpe)
BORDER_WIDTH = 1
ELEMENT_SPACING = 0.8 * cm   # Espace entre les éléments (texte, image) et les bords de la carte

# Couleurs (recto) selon le nom du fichier
COLOR_MAP = {
    "bleu": colors.HexColor("#2D6CDF"),
    "rouge": colors.HexColor("#D64541"),
    "rose": colors.HexColor("#E85D9E"),
    "vert": colors.HexColor("#2ECC71"),
    "jaune": colors.HexColor("#F1C40F"),
}

def pick_color_from_filename(filename: str) -> Tuple[str, colors.Color]:
    low = filename.lower()
    for key in ["bleu", "rouge", "rose", "vert", "jaune"]:
        if key in low:
            return key, COLOR_MAP[key]
    return "bleu", COLOR_MAP["bleu"]

def parse_color_string(color_str: str, default_color: colors.Color) -> colors.Color:
    if not color_str:
        return default_color

    # Try to match predefined color names
    if color_str.lower() in COLOR_MAP:
        return COLOR_MAP[color_str.lower()]

    # Try to match hexadecimal color codes (3 or 6 digits)
    hex_match = re.match(r'^#?([0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?)$', color_str.strip())
    if hex_match:
        # Ensure it's a 6-digit hex code for ReportLab
        hex_code_val = hex_match.group(1)
        if len(hex_code_val) == 3:
            hex_code_val = ''.join([c*2 for c in hex_code_val]) # Expand 3-digit to 6-digit
        hex_code = "#" + hex_code_val
        try:
            return colors.HexColor(hex_code)
        except Exception:
            # Fallback if HexColor parsing fails
            pass
            
    # If neither, return default color
    return default_color

def is_dark(c: colors.Color) -> bool:
    r, g, b = c.red, c.green, c.blue
    lum = 0.2126*r + 0.7152*g + 0.0722*b
    return lum < 0.55

def sniff_dialect(data: str) -> csv.Dialect:
    # Try semicolon first
    if ';' in data:
        try:
            # Check if semicolon works as a reasonable delimiter (e.g., more than one field)
            test_reader = csv.reader(io.StringIO(data), delimiter=';')
            # Check if any row has more than one field, or if there's a header with semicolons
            first_few_lines = data.splitlines()[:5] # Check first 5 lines for consistency
            if any(len(row) > 1 for row in test_reader) or all(';' in line for line in first_few_lines if line.strip()):
                class SemicolonDialect(csv.excel):
                    delimiter = ';'
                return SemicolonDialect()
        except Exception:
            pass # Fall through to other options if this fails

    # Try comma
    if ',' in data:
        try:
            test_reader = csv.reader(io.StringIO(data), delimiter=',')
            if any(len(row) > 1 for row in test_reader):
                class CommaDialect(csv.excel):
                    delimiter = ','
                return CommaDialect()
        except Exception:
            pass # Fall through

    # Try tab
    if '\t' in data:
        try:
            test_reader = csv.reader(io.StringIO(data), delimiter='\t')
            if any(len(row) > 1 for row in test_reader):
                class TabDialect(csv.excel):
                    delimiter = '\t'
                return TabDialect()
        except Exception:
            pass # Fall through

    # Fallback to default Sniffer behavior (which often defaults to comma)
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(data[:4096]) # Use default delimiters for sniffer
        return dialect
    except Exception:
        return csv.get_dialect("excel") # Default to excel dialect (comma-separated by default for excel)

def normalize_header(h: str) -> str:
    return re.sub(r"\s+", "", (h or "").strip().lower())

def read_cards_from_csv(csv_file_content: str) -> List[Dict[str, str]]:
    """
    CSV attendu (souple) :
    - question : colonne 'question' (ou 1re colonne si pas d'en-tête)
    - texte verso : colonne 'texte' / 'reponse' / 'r\u00e9ponse' / 'answer' (ou 2e/3e colonne selon pr\u00e9sence d'en-tête)
    - image recto : colonne 'image_recto' / 'imagerecto' (ou 3e colonne si pas d'en-tête)
    """
    # Use io.StringIO to treat the string content as a file
    f = io.StringIO(csv_file_content)

    dialect = sniff_dialect(csv_file_content)
    reader = csv.reader(f, dialect)
    rows = list(reader)
    if not rows:
        return []

    first = rows[0]
    norm_first = [normalize_header(x) for x in first]
    has_header = any(x in ("question","q","texte","text","reponse","r\u00e9ponse","answer","reponseverso","verso", "image_recto", "imagerecto") for x in norm_first)

    def get_field(d: Dict[str,str], keys: List[str], fallback: str="") -> str:
        for k in keys:
            nk = normalize_header(k)
            for kk, vv in d.items():
                if normalize_header(kk) == nk:
                    return (vv or "").strip()
        return fallback

    out = []
    if has_header:
        headers = norm_first
        for r in rows[1:]:
            if not any(str(x).strip() for x in r):
                continue
            d = {headers[i]: (r[i].strip() if i < len(r) else "") for i in range(len(headers))}
            q_raw = get_field(d, ["question","q"])
            card_color_string = None
            question_text = q_raw # Default to raw question

            # Try to find (color_name_or_hex) at the BEGINNING of the string, case-insensitive
            # Removed ;? from regex as it's a delimiter and shouldn't be in q_raw
            match_beginning = re.match(r'^\s*\(([^)]+)\)\s*(.*)$', q_raw, re.IGNORECASE)
            if match_beginning:
                card_color_string = match_beginning.group(1).strip()
                question_text = match_beginning.group(2).strip()
            else:
                # If not at the beginning, try to find (color_name_or_hex) at the END of the string
                match_end = re.search(r'\s*\(([^)]+)\)\s*$', q_raw, re.IGNORECASE)
                if match_end:
                    card_color_string = match_end.group(1).strip()
                    question_text = re.sub(r'\s*\(([^)]+)\)\s*$', '', q_raw, flags=re.IGNORECASE).strip()

            txt = get_field(d, ["texte","text","reponse","r\u00e9ponse","answer","verso","reponseverso"])
            card_image_recto = get_field(d, ["image_recto", "imagerecto"])
            out.append({"question": question_text, "texte": txt, "card_color_key": card_color_string, "image_recto": card_image_recto})
    else:
        # Sans en-tête : col1=question, col2=texte, col3=image_recto (si pr\u00e9sente)
        for r in rows:
            if not any(str(x).strip() for x in r):
                continue
            q_raw = (r[0].strip() if len(r) > 0 else "")
            card_color_string = None
            question_text = q_raw

            # Try to find (color_name_or_hex) at the BEGINNING of the string, case-insensitive
            # Removed ;? from regex as it's a delimiter and shouldn't be in q_raw
            match_beginning = re.match(r'^\s*\(([^)]+)\)\s*(.*)$', q_raw, re.IGNORECASE)
            if match_beginning:
                card_color_string = match_beginning.group(1).strip()
                question_text = match_beginning.group(2).strip()
            else:
                # If not at the beginning, try to find (color_name_or_hex) at the END of the string
                match_end = re.search(r'\s*\(([^)]+)\)\s*$', q_raw, re.IGNORECASE)
                if match_end:
                    card_color_string = match_end.group(1).strip()
                    question_text = re.sub(r'\s*\(([^)]+)\)\s*$', '', q_raw, flags=re.IGNORECASE).strip()

            txt = (r[1].strip() if len(r) > 1 else "") # Second column for verso text
            card_image_recto = (r[2].strip() if len(r) > 2 else "") # Third column for recto image
            out.append({"question": question_text, "texte": txt, "card_color_key": card_color_string, "image_recto": card_image_recto})

    return out

# ----------------------------
# Mise en page
# ----------------------------
class Grid:
    def __init__(self, page_w, page_h, card_w, card_h, x0, y0):
        self.page_w = page_w
        self.page_h = page_h
        self.card_w = card_w
        self.card_h = card_h
        self.x0 = x0
        self.y0 = y0

def compute_grid() -> Grid:
    page_w, page_h = A4
    usable_w = page_w - 2*MARGIN - (COLS-1)*GAP
    usable_h = page_h - 2*MARGIN - (ROWS-1)*GAP
    card_w = usable_w / COLS
    card_h = usable_h / ROWS
    return Grid(page_w, page_h, card_w, card_h, MARGIN, MARGIN)

def card_xy(grid: Grid, col: int, row: int) -> Tuple[float,float]:
    # row 0 en haut
    x = grid.x0 + col*(grid.card_w + GAP)
    y_top = grid.page_h - grid.y0 - row*(grid.card_h + GAP)
    y = y_top - grid.card_h
    return x, y

def draw_card_border(c: canvas.Canvas, x: float, y: float, w: float, h: float, stroke_color=colors.lightgrey):
    c.setLineWidth(BORDER_WIDTH)
    c.setStrokeColor(stroke_color)
    c.rect(x, y, w, h, stroke=1, fill=0)

def draw_centered_text_in_box(c: canvas.Canvas, x: float, y: float, w: float, h: float, text: str, style: ParagraphStyle):
    pad = 6 # Internal padding for the text within the card

    # Calculate the inner dimensions for the text area
    inner_x = x + pad
    inner_y = y + pad
    inner_w = w - 2 * pad
    inner_h = h - 2 * pad

    # Replace newlines with line breaks for display, keeping semicolons as-is
    formatted_text = (text or "").replace("\n","<br/>")
    p = Paragraph(formatted_text if formatted_text.strip() else "&nbsp;", style)

    # Get the actual height the paragraph would take if wrapped within inner_w
    # We pass a temporary canvas and a very large height to allow it to compute its natural height
    text_width, text_height = p.wrapOn(c, inner_w, inner_h * 100)

    # Ensure text_height does not exceed inner_h, and shrink if necessary
    if text_height > inner_h:
        text_height = inner_h

    # Calculate vertical offset to center the text
    y_offset = (inner_h - text_height) / 2

    # Draw the paragraph
    # The y-coordinate for drawOn is the bottom-left corner of the paragraph.
    # We want to place the bottom of the paragraph at (inner_y + y_offset).
    p.drawOn(c, inner_x, inner_y + y_offset)

def build_pdf(cards: List[Dict[str,str]], default_back_color: colors.Color, output_buffer: io.BytesIO, uploaded_recto_images: Dict[str, Image.Image] = None):
    grid = compute_grid()

    base_font = "Helvetica"
    style_verso = ParagraphStyle(
        "Verso", fontName=base_font, fontSize=12.5, leading=14.5,
        alignment=TA_CENTER, textColor=colors.black
    )

    cards10 = (cards[:NB_CARTES] + [{"question":"","texte":""}] * NB_CARTES)[:NB_CARTES]

    c = canvas.Canvas(output_buffer, pagesize=A4)

    temp_image_files_to_clean = [] # List to keep track of temporary files for cleanup

    # -------- Recto --------
    for i in range(NB_CARTES):
        row = i // COLS
        col = i % COLS
        x, y = card_xy(grid, col, row)

        # Determine the background color for the current card
        card_specific_color_string = cards10[i].get("card_color_key")
        current_back_color = parse_color_string(card_specific_color_string, default_back_color)

        # Recto text style: color adapted to background (depends on current_back_color)
        style_recto = ParagraphStyle(
            "Recto", fontName=base_font, fontSize=16, leading=18,
            alignment=TA_CENTER, textColor=(colors.white if is_dark(current_back_color) else colors.black)
        )

        # Fill recto with background color
        c.setFillColor(current_back_color)
        c.rect(x, y, grid.card_w, grid.card_h, stroke=0, fill=1)

        question_text_for_card = cards10[i].get("question", "").strip()
        card_recto_image_filename = cards10[i].get("image_recto", "").strip()

        current_recto_pil_image = None
        if card_recto_image_filename and uploaded_recto_images and card_recto_image_filename in uploaded_recto_images:
             current_recto_pil_image = uploaded_recto_images[card_recto_image_filename]

        image_to_draw_path = None
        if current_recto_pil_image:
            try:
                r = current_back_color.red
                g = current_back_color.green
                b = current_back_color.blue
                bg_color_tuple = (int(r * 255), int(g * 255), int(b * 255))

                # Create a new RGB image with the desired background color
                alpha_composite_img = Image.new('RGB', current_recto_pil_image.size, bg_color_tuple)
                alpha_composite_img.paste(current_recto_pil_image, (0, 0), current_recto_pil_image) # The original_pil_image is used as the mask for pasting

                # Save the composited RGB image to a temporary file on disk
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_png_file:
                    image_to_draw_path = temp_png_file.name
                    alpha_composite_img.save(temp_png_file, format='PNG')
                temp_image_files_to_clean.append(image_to_draw_path) # Add to cleanup list

            except Exception as e:
                st.error(f"Erreur lors du compositing de l'image de recto pour la carte {i}: {e}")
                image_to_draw_path = None


        if image_to_draw_path:
            if not question_text_for_card:
                # No text, image takes up 90% of card height, centered
                img_h = 0.9 * grid.card_h
                img_w = img_h # 1:1 aspect ratio

                # Calculate positions based on ELEMENT_SPACING
                img_x = x + (grid.card_w - img_w) / 2 # Centered horizontally
                img_y = y + (grid.card_h - img_h) / 2 # Center vertically

                try:
                    c.drawImage(image_to_draw_path, img_x, img_y,
                                width=img_w, height=img_h, preserveAspectRatio=True)
                except Exception as e:
                    st.error(f"Erreur lors du dessin de l'image (90% hauteur, sans texte) : {e}")
                    # If image drawing fails, draw empty text centrally as a fallback
                    draw_centered_text_in_box(c, x, y, grid.card_w, grid.card_h, "", style_recto)
            else:
                # Text is present, use original image/text layout
                img_h = grid.card_h / 2
                img_w = img_h

                img_x = x + (grid.card_w - img_w) / 2
                img_y = y + ELEMENT_SPACING

                text_box_h = grid.card_h - (3 * ELEMENT_SPACING + img_h)

                text_box_x = x
                text_box_y = img_y + img_h + ELEMENT_SPACING
                text_box_w = grid.card_w

                try:
                    c.drawImage(image_to_draw_path, img_x, img_y,
                                width=img_w, height=img_h, preserveAspectRatio=True)
                except Exception as e:
                    st.error(f"Erreur lors du dessin de l'image (avec texte) : {e}")
                    # Fallback: draw text in full card area if image drawing still fails
                    draw_centered_text_in_box(c, x, y, grid.card_w, grid.card_h, question_text_for_card, style_recto)
                    continue

                draw_centered_text_in_box(c, text_box_x, text_box_y, text_box_w, text_box_h, question_text_for_card, style_recto)
        else:
            # No image or image processing failed, draw text in the full card area
            draw_centered_text_in_box(c, x, y, grid.card_w, grid.card_h, question_text_for_card, style_recto)

    c.showPage()

    # -------- Verso (colonnes inversées) --------
    for i in range(NB_CARTES):
        row = i // COLS
        col = i % COLS
        back_col = (COLS - 1 - col) # inversion colonnes pour impression recto/verso
        x, y = card_xy(grid, back_col, row)

        # Verso now only draws text, removed all image logic
        draw_centered_text_in_box(c, x, y, grid.card_w, grid.card_h, cards10[i].get("texte", ""), style_verso)

    c.save()

    # Cleanup temporary files created during image processing
    for temp_file in temp_image_files_to_clean:
        try:
            os.remove(temp_file)
        except OSError as e:
            st.warning(f"Erreur lors de la suppression du fichier temporaire {temp_file}: {e}")


# ----------------------------
# Streamlit Application Logic
# ----------------------------
st.title("Générateur de cartes à tout faire")

st.write("Uploadez votre fichier CSV et un fichier ZIP d'images (facultatif) pour générer 10 cartes recto/verso sur une feuille A4 pdf.")
st.text("Le contenu du fichier CSV est constitu\u00e9e au maximum de 10 lignes du type :")
st.text("ma question1 (couleur_ou_#CODEHEX) ; ma r\u00e9ponse1 ; mon_image_recto.png")
st.text("ma question2 (couleur_ou_#CODEHEX) ; ma r\u00e9ponse2")
st.text("etc.")
st.write("(couleur_ou_#CODEHEX) est la couleur du recto de la carte - choix possibles : bleu, rouge, rose, vert, jaune ou un code hexad\u00e9cimal comme #FF00FF ou #F00. ")
st.write("Si aucune couleur n'est indiqu\u00e9e (maquestion1 ; mar\u00e9ponse1) alors la couleur par d\u00e9faut du recto est le bleu.")
st.write("Le nom du fichier image dans la 3e colonne du CSV doit correspondre exactement au nom d'un fichier PNG/JPG dans le ZIP d'images recto.")

# CSV Upload
uploaded_csv_file = st.file_uploader("Uploader le fichier CSV", type=["csv"])

# Image Upload for Recto (multiple images via ZIP)
uploaded_recto_images_zip = st.file_uploader("Uploader un fichier ZIP d'images PNG/JPG (facultatif) pour les rectos", type=["zip"])

recto_images_dict = {}
if uploaded_recto_images_zip:
    st.info("D\u00e9compression des images de recto...")
    with tempfile.TemporaryDirectory() as tempdir:
        with zipfile.ZipFile(uploaded_recto_images_zip, 'r') as zip_ref:
            zip_ref.extractall(tempdir)
        for filename in os.listdir(tempdir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(tempdir, filename)
                try:
                    img = Image.open(filepath).convert('RGBA') # Convert to RGBA for consistent handling
                    recto_images_dict[filename] = img
                except Exception as e:
                    st.warning(f"Impossible de charger l'image de recto {filename}: {e}")
    if recto_images_dict:
        st.success(f"{len(recto_images_dict)} images de recto charg\u00e9es depuis le fichier ZIP.")
    else:
        st.warning("Aucune image valide trouv\u00e9e dans le fichier ZIP des images de recto.")


if uploaded_csv_file is None:
    st.warning("Veuillez uploader un fichier CSV pour commencer.")
elif uploaded_csv_file is not None:
    # Read CSV content from the uploaded file
    csv_content = uploaded_csv_file.getvalue().decode("utf-8")
    csv_name = uploaded_csv_file.name

    color_name_from_filename, default_back_color = pick_color_from_filename(csv_name)
    st.info(f"Couleur par d\u00e9faut d\u00e9tect\u00e9e (via nom de fichier) : {color_name_from_filename}")

    cards = read_cards_from_csv(csv_content)

    st.info(f"Lignes lues : {len(cards)} (on utilise les {NB_CARTES} premi\u00e8res)")

    if st.button("G\u00e9n\u00e9rer le PDF"):
        if cards:
            output_buffer = io.BytesIO()
            # Pass the dictionary of recto images to build_pdf
            build_pdf(cards, default_back_color, output_buffer, uploaded_recto_images=recto_images_dict)

            st.success(f"PDF g\u00e9n\u00e9r\u00e9 : {OUTPUT_PDF}")
            st.download_button(
                label="T\u00e9l\u00e9charger le PDF",
                data=output_buffer.getvalue(),
                file_name=OUTPUT_PDF,
                mime="application/pdf"
            )
        else:
            st.error("Aucune carte n'a pu \u00eatre lue depuis le fichier CSV. La g\u00e9n\u00e9ration du PDF est annul\u00e9e.")
