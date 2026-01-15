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
CUT_MARK_LENGTH = 0.5 * cm   # Longueur des traits de coupe

# Couleurs (verso) selon le nom du fichier
COLOR_MAP = {
    "bleu": colors.HexColor("#2D6CDF"),
    "rouge": colors.HexColor("#D64541"),
    "rose": colors.HexColor("#E85D9E"),
    "vert": colors.HexColor("#2ECC71"),
    "jaune": colors.HexColor("#F1C40F"),
    "blanc": colors.white, # Ajout du blanc
}

def pick_color_from_filename(filename: str) -> Tuple[str, colors.Color]:
    low = filename.lower()
    for key in ["bleu", "rouge", "rose", "vert", "jaune", "blanc"]:
        if key in low:
            return key, COLOR_MAP[key]
    return "bleu", COLOR_MAP["bleu"]

def is_dark(c: colors.Color) -> bool:
    r, g, b = c.red, c.green, c.blue
    # For white, we need to explicitly return false for dark to ensure black text
    if c == colors.white:
        return False
    lum = 0.2126*r + 0.7152*g + 0.0722*b
    return lum < 0.55

def sniff_dialect(data: str) -> csv.Dialect:
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(data[:4096], delimiters=";,|, ,\t,")
    except Exception:
        dialect = csv.get_dialect("excel")
    return dialect

def normalize_header(h: str) -> str:
    return re.sub(r"\s+", "", (h or "").strip().lower())

def read_cards_from_csv(csv_file_content: str) -> List[Dict[str, str]]:
    """
    CSV attendu (souple) :
    - question : colonne 'question' (ou 1re colonne si pas d'en-tête)
    - texte verso : colonne 'texte' / 'reponse' / 'réponse' / 'answer' (ou 2e/3e colonne selon présence d'en-tête)
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
    has_header = any(x in ("question","q","texte","text","reponse","réponse","answer","reponseverso","verso", "image_recto", "imagerecto") for x in norm_first)

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
            card_color_key = None
            question_text = q_raw # Default to raw question

            # Regex to find (color) at the end of the string, case-insensitive
            match = re.search(r'\(([^)]+)\)\s*$', q_raw, re.IGNORECASE)
            if match:
                extracted_color_name = match.group(1).lower().strip()
                if extracted_color_name in COLOR_MAP:
                    card_color_key = extracted_color_name
                    question_text = re.sub(r'\s*\(([^)]+)\)\s*$', '', q_raw, flags=re.IGNORECASE).strip()

            txt = get_field(d, ["texte","text","reponse","réponse","answer","verso","reponseverso"])
            card_image_recto = get_field(d, ["image_recto", "imagerecto"])
            out.append({"question": question_text, "texte": txt, "card_color_key": card_color_key, "image_recto": card_image_recto})
    else:
        # Sans en-tête : col1=question, col2=texte, col3=image_recto (si présente)
        for r in rows:
            if not any(str(x).strip() for x in r):
                continue
            q_raw = (r[0].strip() if len(r) > 0 else "")
            card_color_key = None
            question_text = q_raw

            match = re.search(r'\(([^)]+)\)\s*$', q_raw, re.IGNORECASE)
            if match:
                extracted_color_name = match.group(1).lower().strip()
                if extracted_color_name in COLOR_MAP:
                    card_color_key = extracted_color_name
                    question_text = re.sub(r'\s*\(([^)]+)\)\s*$', '', q_raw, flags=re.IGNORECASE).strip()

            txt = (r[1].strip() if len(r) > 1 else "") # Second column for verso text
            card_image_recto = (r[2].strip() if len(r) > 2 else "") # Third column for recto image
            out.append({"question": question_text, "texte": txt, "card_color_key": card_color_key, "image_recto": card_image_recto})

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

    p = Paragraph((text or "").replace("\n","<br/>") if (text or "").strip() else "&nbsp;", style)

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

def draw_cut_marks(c: canvas.Canvas, grid: Grid):
    c.setLineWidth(0.2) # Thinner lines for cut marks
    c.setStrokeColor(colors.black)

    # Vertical cut marks
    for col in range(COLS):
        x_left = grid.x0 + col * (grid.card_w + GAP)
        x_right = x_left + grid.card_w

        # Top marks
        c.line(x_left, grid.page_h - MARGIN, x_left, grid.page_h - MARGIN - CUT_MARK_LENGTH)
        c.line(x_right, grid.page_h - MARGIN, x_right, grid.page_h - MARGIN - CUT_MARK_LENGTH)
        
        # Bottom marks
        c.line(x_left, MARGIN, x_left, MARGIN + CUT_MARK_LENGTH)
        c.line(x_right, MARGIN, x_right, MARGIN + CUT_MARK_LENGTH)

        # Marks in the middle horizontal gap
        if col < COLS - 1:
            x_mid = x_right + GAP / 2
            # Draw vertical lines in the middle of the gap, from top margin to bottom margin
            c.line(x_mid, grid.page_h - MARGIN, x_mid, MARGIN)

    # Horizontal cut marks
    for row in range(ROWS):
        y_bottom = MARGIN + row * (grid.card_h + GAP)
        y_top = y_bottom + grid.card_h

        # Left marks
        c.line(MARGIN, y_bottom, MARGIN + CUT_MARK_LENGTH, y_bottom)
        c.line(MARGIN, y_top, MARGIN + CUT_MARK_LENGTH, y_top)

        # Right marks
        c.line(grid.page_w - MARGIN, y_bottom, grid.page_w - MARGIN - CUT_MARK_LENGTH, y_bottom)
        c.line(grid.page_w - MARGIN, y_top, grid.page_w - MARGIN - CUT_MARK_LENGTH, y_top)

        # Marks in the middle vertical gap
        if row < ROWS - 1:
            y_mid = y_top + GAP / 2
            # Draw horizontal lines in the middle of the gap, from left margin to right margin
            c.line(MARGIN, y_mid, grid.page_w - MARGIN, y_mid)

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
        card_specific_color_key = cards10[i].get("card_color_key")
        current_back_color = COLOR_MAP.get(card_specific_color_key, default_back_color)

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

    draw_cut_marks(c, grid)
    c.showPage()

    # -------- Verso (colonnes inversées) --------
    for i in range(NB_CARTES):
        row = i // COLS
        col = i % COLS
        back_col = (COLS - 1 - col) # inversion colonnes pour impression recto/verso
        x, y = card_xy(grid, back_col, row)

        # Verso now only draws text, removed all image logic
        draw_centered_text_in_box(c, x, y, grid.card_w, grid.card_h, cards10[i].get("texte", ""), style_verso)
    
    # Removed draw_cut_marks for verso page as requested.
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
st.text("Le contenu du fichier CSV est constituée au maximum de 10 lignes du type :")
st.text("ma question1 (couleur) ; ma réponse1 ; mon_image_recto.png")
st.text("ma question2 (couleur) ; ma réponse2")
st.text("etc.")
st.write("(couleur) est la couleur du recto de la carte - choix possibles : bleu, rouge, rose, vert, jaune, blanc. ")
st.write("Si aucune couleur n'est indiquée (maquestion1 ; maréponse1) alors la couleur par défaut du recto est le bleu.")
st.write("Le nom du fichier image dans la 3e colonne du CSV doit correspondre exactement au nom d'un fichier PNG/JPG dans le ZIP d'images recto.")

# CSV Upload
uploaded_csv_file = st.file_uploader("Uploader le fichier CSV", type=["csv"])

# Image Upload for Recto (multiple images via ZIP)
uploaded_recto_images_zip = st.file_uploader("Uploader un fichier ZIP d'images PNG/JPG (facultatif) pour les rectos", type=["zip"])

recto_images_dict = {}
if uploaded_recto_images_zip:
    st.info("Décompression des images de recto...")
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
        st.success(f"{len(recto_images_dict)} images de recto chargées depuis le fichier ZIP.")
    else:
        st.warning("Aucune image valide trouvée dans le fichier ZIP des images de recto.")


if uploaded_csv_file is None:
    st.warning("Veuillez uploader un fichier CSV pour commencer.")
elif uploaded_csv_file is not None:
    # Read CSV content from the uploaded file
    csv_content = uploaded_csv_file.getvalue().decode("utf-8")
    csv_name = uploaded_csv_file.name

    color_name, default_back_color = pick_color_from_filename(csv_name)
    st.info(f"Couleur par défaut détectée (via nom de fichier) : {color_name}")

    cards = read_cards_from_csv(csv_content)
    st.info(f"Lignes lues : {len(cards)} (on utilise les {NB_CARTES} premières)")

    if st.button("Générer le PDF"):
        if cards:
            output_buffer = io.BytesIO()
            # Pass the dictionary of recto images to build_pdf
            build_pdf(cards, default_back_color, output_buffer, uploaded_recto_images=recto_images_dict)

            st.success(f"PDF généré : {OUTPUT_PDF}")
            st.download_button(
                label="Télécharger le PDF",
                data=output_buffer.getvalue(),
                file_name=OUTPUT_PDF,
                mime="application/pdf"
            )
        else:
            st.error("Aucune carte n'a pu être lue depuis le fichier CSV. La génération du PDF est annulée.")
