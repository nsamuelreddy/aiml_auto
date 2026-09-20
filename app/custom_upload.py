import base64
import io
import json

import streamlit as st


class UploadedFileLike:
    def __init__(self, name, data, content_type="application/octet-stream"):
        self.name = name
        self.size = len(data)
        self.type = content_type
        self._buffer = io.BytesIO(data)

    def read(self, size=-1):
        return self._buffer.read(size)

    def seek(self, position=0):
        return self._buffer.seek(position)

    def tell(self):
        return self._buffer.tell()


def render_custom_upload(label="Upload", accepted_types=".csv,.xlsx,.xls,.json", max_size_mb=10):
    """
    Renders a fully self-contained modern upload card (title + description + drop zone)
    as a single HTML component. Returns an UploadedFileLike or None.
    """
    component_html = f"""
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        html, body {{
            height: 100%;
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
            background: transparent;
            color: #f8fafc;
        }}
        .card {{
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: space-between;
            gap: 24px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 22px 26px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.04);
            min-height: 140px;
        }}
        .card-info {{
            flex: 1 1 auto;
            min-width: 0;
        }}
        .card-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #f8fafc;
            letter-spacing: -0.01em;
            margin-bottom: 8px;
        }}
        .card-sub {{
            font-size: 0.875rem;
            color: #94a3b8;
            line-height: 1.65;
        }}
        .upload-zone {{
            flex: 0 0 auto;
            width: 210px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 7px;
            border: 1.5px dashed rgba(99, 179, 237, 0.38);
            border-radius: 14px;
            padding: 18px 16px;
            cursor: pointer;
            background: rgba(56, 120, 220, 0.07);
            transition: all 0.2s ease;
            text-align: center;
            min-height: 96px;
            position: relative;
        }}
        .upload-zone:hover, .upload-zone.drag-over {{
            border-color: rgba(99, 179, 237, 0.8);
            background: rgba(56, 120, 220, 0.15);
            box-shadow: 0 0 0 3px rgba(99, 179, 237, 0.13);
            transform: translateY(-1px);
        }}
        .upload-icon {{
            font-size: 1.6rem;
            line-height: 1;
        }}
        .upload-label-text {{
            font-size: 0.88rem;
            font-weight: 700;
            color: #7dd3fc;
        }}
        .upload-hint {{
            font-size: 0.7rem;
            color: #64748b;
            line-height: 1.4;
        }}
        .file-badge {{
            display: none;
            font-size: 0.72rem;
            color: #4ade80;
            font-weight: 600;
            margin-top: 2px;
            word-break: break-all;
            max-width: 190px;
        }}
        input[type="file"] {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }}
    </style>

    <div class="card">
        <div class="card-info">
            <div class="card-title">📂 Upload Dataset</div>
            <div class="card-sub">
                Drop a file or click the zone to browse.<br>
                Supports <strong style="color:#cbd5e1">CSV, Excel (.xlsx/.xls), JSON</strong>.<br>
                Maximum file size: {max_size_mb} MB.
            </div>
        </div>
        <div class="upload-zone" id="upload-zone">
            <input type="file" id="upload-input" accept="{accepted_types}" />
            <div class="upload-icon">☁️</div>
            <div class="upload-label-text">Choose File</div>
            <div class="upload-hint">{accepted_types.replace(",", " · ")}</div>
            <div class="file-badge" id="file-name"></div>
        </div>
    </div>

    <script>
        const input    = document.getElementById('upload-input');
        const zone     = document.getElementById('upload-zone');
        const badge    = document.getElementById('file-name');

        zone.addEventListener('dragover',  e => {{ e.preventDefault(); zone.classList.add('drag-over'); }});
        zone.addEventListener('dragleave', ()  => zone.classList.remove('drag-over'));
        zone.addEventListener('drop',      e  => {{
            e.preventDefault();
            zone.classList.remove('drag-over');
            handleFile(e.dataTransfer.files[0]);
        }});
        input.addEventListener('change', e => handleFile(e.target.files[0]));

        function handleFile(file) {{
            if (!file) {{ Streamlit.setComponentValue(null); return; }}
            badge.textContent = '✓ ' + file.name;
            badge.style.display = 'block';
            const reader = new FileReader();
            reader.onload = function () {{
                Streamlit.setComponentValue(JSON.stringify({{
                    name: file.name,
                    type: file.type || 'application/octet-stream',
                    data: reader.result.split(',')[1],
                    size: file.size
                }}));
            }};
            reader.readAsDataURL(file);
        }}
    </script>
    """

    component_value = st.components.v1.html(component_html, height=168)
    if not component_value:
        return None

    try:
        payload = json.loads(component_value)
    except Exception:
        return None

    if not payload or not payload.get("name"):
        return None

    content = base64.b64decode(payload["data"])
    return UploadedFileLike(payload["name"], content, payload.get("type", "application/octet-stream"))

