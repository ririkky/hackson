import os
import csv
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

# =====================================================
# output/file/results.csv を読み込み、画像と評価を対応づけ
# =====================================================
def load_results_from_csv():
    # app3.py（UI直下）から見たパス
    base_dir = os.path.join(app.root_path, 'output')
    csv_path = os.path.join(base_dir, 'file', 'results.csv')
    images_dir = os.path.join(base_dir, 'images')

    image_map = []

    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)

            for row in reader:
                mesh_id = row.get('メッシュID', '').strip()
                filename = row.get('画像', '').strip()
                evaluation = row.get('評価', '').strip()

                # images内のファイルを確認
                image_path = os.path.join(images_dir, filename)
                if not os.path.exists(image_path):
                    print(f"⚠️ 画像が見つかりません: {image_path}")
                    continue

                # FlaskでアクセスできるURLを作成
                image_url = f"/images/{filename}"

                image_map.append({
                    'mesh_id': mesh_id,
                    'filename': filename,
                    'url': image_url,
                    'evaluation': evaluation
                })

    except FileNotFoundError:
        print(f"⚠️ {csv_path} が見つかりません。")
    except Exception as e:
        print(f"⚠️ CSV読み込みエラー: {e}")

    return image_map


# =====================================================
# Flaskルート
# =====================================================
@app.route('/')
def index():
    image_data = load_results_from_csv()
    return render_template('index.html', images=image_data)


@app.route('/api/images')
def get_images():
    image_data = load_results_from_csv()
    return jsonify(image_data)


# =====================================================
# /images/... にアクセスしたら output/images から返す
# =====================================================
@app.route('/images/<path:filename>')
def serve_images(filename):
    images_dir = os.path.join(app.root_path, 'output', 'images')
    return send_from_directory(images_dir, filename)


if __name__ == '__main__':
    app.run(debug=True)
