from flask import Flask, request, send_file
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import tempfile
import os

app = Flask(__name__)

# Voigt関数の定義
def voigt(x, amplitude, center, sigma, gamma):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z))

# 曲線の最大値を求める関数
def find_peak(x, y):
    max_index = np.argmax(y)
    return x[max_index], y[max_index]

# FWHMを計算する関数
def calculate_fwhm(x, y, background):
    max_x, max_y = find_peak(x, y)
    half_max = max_y / 2.0
    half_max_value = half_max + background / 2.0

    indices = np.where(y >= half_max_value)[0]
    if len(indices) < 2:
        return None, None  

    x1, x2 = x[indices[0]], x[indices[-1]]
    fwhm = x2 - x1

    return fwhm, half_max_value

@app.route('/')
def upload_form():
    return '''
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>XRD Rocking Curve Fitting</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 40px;
                background-color: #f4f4f4;
            }
            h1 {
                color: #333;
            }
            .container {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                display: inline-block;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            button {
                background-color: #007BFF;
                color: white;
                border: none;
                padding: 10px 15px;
                font-size: 16px;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <h1>XRD Rocking Curve Fitting</h1>
        <div class="container">
            <form action="/upload" method="post" enctype="multipart/form-data">
                <p>Upload your .txt file (XRD Data):</p>
                <input type="file" name="file" id="file"><br>
                <button type="submit">Submit</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file_with_fit():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    try:
        # データの読み込み
        data = np.loadtxt(file)
        x_data, y_data = data[:, 0], data[:, 1]

        # 最大値の x を 0 に補正
        peak_x, peak_y = find_peak(x_data, y_data)
        x_data -= peak_x  

        # 最初の10点のyの平均をバックグラウンドとして求める
        background = np.mean(y_data[:10])

        # ピーク周辺のデータのみを使用
        fit_range = 1.0  
        mask = (x_data > -fit_range) & (x_data < fit_range)
        x_fit, y_fit = x_data[mask], y_data[mask]

        # 初期パラメータの設定
        initial_guess = [peak_y - background, 0, np.std(x_fit) / 2, np.std(x_fit) / 2]

        # Voigt関数のフィッティング
        params, covariance = curve_fit(voigt, x_fit, y_fit, p0=initial_guess, maxfev=50000)

        # フィットデータ生成
        x_data_extended = np.linspace(min(x_fit), max(x_fit), 1000)
        fitted_y = voigt(x_data_extended, *params)

        # FWHMおよびHalf Maximumの計算
        fwhm, half_max_value = calculate_fwhm(x_data_extended, fitted_y, background)
        if fwhm is None:
            return "FWHM could not be calculated", 500

        # フィットデータの保存
        fit_data = np.column_stack((x_data_extended, fitted_y))
        temp_dir = tempfile.gettempdir()
        fit_file_path = os.path.join(temp_dir, "fitted_data.txt")
        np.savetxt(fit_file_path, fit_data, header="X\tFitted_Y", fmt="%.6f")

        # プロットの作成
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, 'bo', label='Data')
        plt.plot(x_data_extended, fitted_y, 'r-', label='Fit')
        plt.axhline(y=half_max_value, color='gray', linestyle='--', label='Half Max')
        plt.axvline(x=params[1] - fwhm / 2, color='green', linestyle='--', label='FWHM')
        plt.axvline(x=params[1] + fwhm / 2, color='green', linestyle='--')
        plt.legend()
        plt.xlabel("X-axis (Shifted)")
        plt.ylabel("Y-axis")
        plt.title("Voigt Function Fitting")

        # グラフを保存してブラウザで表示
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return f'''
        <h1>XRD Rocking Curve Fitting Result</h1>
        <img src="data:image/png;base64,{image_base64}" /><br>
        <p><b>FWHM: {fwhm:.4f}</b></p>
        <a href="/download"><button>Download Fitted Data</button></a>
        '''

    except Exception as e:
        return f"Error during processing: {str(e)}", 500

@app.route('/download')
def download_file():
    temp_dir = tempfile.gettempdir()
    fit_file_path = os.path.join(temp_dir, "fitted_data.txt")
    if not os.path.exists(fit_file_path):
        return "File not found", 404
    return send_file(fit_file_path, as_attachment=True, download_name="fitted_data.txt")

if __name__ == '__main__':
    app.run(debug=True)
