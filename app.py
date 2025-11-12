from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import io
import os

app = Flask(__name__, static_folder='assets')

# Files
MODEL_FILE = "concrete_mix_model_checked.joblib"
ENC_FILE = "mix_encoders_checked.joblib"
LOGO_FILE = os.path.join(app.static_folder, "smartbuild.png")  # your logo

# Load model + encoders
model_bundle = joblib.load(MODEL_FILE)
encoder_bundle = joblib.load(ENC_FILE)

model = model_bundle["model"]
features = model_bundle.get("features", None)
encoders = encoder_bundle["encoders"]
scaler = encoder_bundle["scaler"]
num_cols = encoder_bundle["num_cols"]

# IS 10262 helpers
def get_factor_x(fck):
    return 5 if fck <= 15 else 6 if fck <= 25 else 6.5 if fck <= 50 else 7

def get_std_dev_s(fck):
    return 3.5 if fck <= 15 else 4 if fck <= 25 else 5 if fck <= 50 else 5.5

curve1 = {20:0.63,25:0.58,30:0.54,35:0.50,40:0.47,45:0.45,50:0.43,55:0.41,60:0.39}
curve2 = {20:0.58,25:0.53,30:0.49,35:0.46,40:0.43,45:0.41,50:0.39,55:0.37,60:0.35,70:0.33,80:0.31,90:0.29,100:0.27}
curve3 = {30:0.44,35:0.42,40:0.40,45:0.38,50:0.36,55:0.34,60:0.33,70:0.31,80:0.29,90:0.27,100:0.25}

def interp_wc(fck, c):
    xs = np.array(sorted(c))
    ys = np.array([c[x] for x in xs])
    return float(np.interp(fck, xs, ys))

def get_wc(fck, ctype):
    if ctype == "OPC33":
        return interp_wc(fck, curve1)
    elif ctype == "OPC53":
        return interp_wc(fck, curve3)
    else:
        return interp_wc(fck, curve2)

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    warnings = []
    if request.method == "POST":
        try:
            fck = float(request.form["fck"])
            ctype = request.form["ctype"]
            agg_type = request.form["agg_type"]
            agg_size = int(request.form["agg_size"])
            fzone = request.form["fzone"]
            exposure = request.form["exposure"]
            slump = float(request.form["slump"])
            admixt = request.form["admixt"]
            admix_dos = float(request.form["admix_dos"])
            min_add = float(request.form["min_add"])
        except Exception as e:
            return render_template("index.html", result=None, warnings=[f"Invalid input: {e}"])

        # Compute IS params
        factor_x = get_factor_x(fck)
        std_s = get_std_dev_s(fck)
        wc_ratio = round(get_wc(fck, ctype), 3)

        # Warnings
        if fck < 10 or fck > 100:
            warnings.append("fck outside typical range (10–100 MPa).")
        if not 0.25 <= wc_ratio <= 0.7:
            warnings.append("Auto w/c ratio outside typical range (0.25–0.7).")
        if slump > 200:
            warnings.append("High slump; may require HR admixture.")
        if min_add > 40:
            warnings.append("High mineral admixture % (>40%).")

        # Prepare input dataframe
        data = {
            "Grade_fck": fck,
            "Cement_Type": ctype,
            "Max_Aggregate_Size_mm": agg_size,
            "Aggregate_Type": agg_type,
            "Fine_Agg_Zone": fzone,
            "Exposure_Condition": exposure,
            "Workability_Slump_mm": slump,
            "Admixture_Type": admixt,
            "Admixture_Dosage_%": admix_dos,
            "Mineral_Admixture_%": min_add,
            "w_c_ratio": wc_ratio,
            "Factor_X": factor_x,
            "Std_Deviation_S": std_s
        }
        X_input = pd.DataFrame([data])

        # Encode categoricals (fallback to 0)
        for col, le in encoders.items():
            if col in X_input:
                try:
                    X_input[col] = le.transform(X_input[col].astype(str))
                except Exception:
                    X_input[col] = [0]

        # Scale numeric
        try:
            X_input[num_cols] = scaler.transform(X_input[num_cols])
        except Exception as e:
            return render_template("index.html", result=None, warnings=[f"Scaling error: {e}"])

        # Predict
        pred = model.predict(X_input)[0]

        if len(pred) == 3:
            cementitious = float(pred[0])
            fine = float(pred[1])
            coarse = float(pred[2])
            water = round(cementitious * wc_ratio, 2)
        elif len(pred) == 4:
            cementitious = float(pred[0])
            water = float(pred[1])
            fine = float(pred[2])
            coarse = float(pred[3])
        else:
            return render_template("index.html", result=None, warnings=["Unexpected model output shape."])

        total = round(cementitious + water + fine + coarse, 2)
        fa_ca = round(fine / (fine + coarse), 2) if (fine + coarse) > 0 else 0.0

        result = {
            "cement": round(cementitious, 2),
            "water": water,
            "fine": round(fine, 2),
            "coarse": round(coarse, 2),
            "wc": wc_ratio,
            "factor_x": factor_x,
            "std_s": std_s,
            "total": total,
            "fa_ca": fa_ca
        }

    return render_template("index.html", result=result, warnings=warnings)

# PDF route (includes college name text; uses your SmartBuild logo)
@app.route("/download-report", methods=["POST"])
def download_report():
    form = request.form.to_dict()
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Draw logo if present
    if os.path.exists(LOGO_FILE):
        try:
            logo = ImageReader(LOGO_FILE)
            pdf.drawImage(logo, 50, height - 110, width=90, height=90, mask='auto')
        except Exception:
            pass

    # Header text
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(160, height - 60, "AI & ML Integrated Concrete Mix Design Report")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(160, height - 78, "Machine Learning Model integrated with IS 10262:2019 guidance")
    pdf.drawString(160, height - 94, f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    # College and developer info
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(50, height - 140, "Developer / Institution:")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(180, height - 140, "Amrit Pal | Civil Engineering | Batch 2024")
    pdf.drawString(180, height - 156, "Dr. B. R. Ambedkar National Institute of Technology Jalandhar")
    pdf.drawString(180, height - 172, "Portfolio: SmartBuild byAmrit")

    # Results table
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, height - 200, "Predicted Mix Proportions (kg/m³)")
    pdf.setFont("Helvetica", 10)
    rows = [
        ("Cementitious Material", form.get("cement", "")),
        ("Water", form.get("water", "")),
        ("Fine Aggregate", form.get("fine", "")),
        ("Coarse Aggregate", form.get("coarse", "")),
        ("Water-Cement Ratio", form.get("wc", "")),
        ("Factor X", form.get("factor_x", "")),
        ("Standard Deviation S", form.get("std_s", "")),
        ("Total Mix Mass", form.get("total", "")),
        ("FA/(FA+CA)", form.get("fa_ca", ""))
    ]
    y = height - 220
    for name, value in rows:
        pdf.drawString(60, y, name)
        pdf.drawRightString(520, y, str(value))
        y -= 18

    pdf.setFont("Helvetica-Oblique", 8)
    pdf.drawString(50, 40, "Generated by SmartBuild byAmrit — AI & ML Integrated Concrete Mix Design Predictor")
    pdf.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="Concrete_Mix_Report.pdf", mimetype='application/pdf')

# About page
@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)

