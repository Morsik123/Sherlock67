"""
ai_analyst.py — AI-powered flight analysis using Claude API.

Sends structured flight metrics and GPS/IMU summaries to Claude
and returns a natural-language mission assessment in Ukrainian.
"""

import json
import urllib.request
import urllib.error
from typing import Optional


def _call_claude(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Call Anthropic Claude API with a prompt.
    API key is read from env var ANTHROPIC_API_KEY if not provided.
    """
    import os
    key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')
    if not key:
        return "⚠️ ANTHROPIC_API_KEY не встановлено. Додайте ключ у змінні середовища."

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode('utf-8')

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data['content'][0]['text']
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='ignore')
        return f"❌ API Error {e.code}: {body[:300]}"
    except Exception as e:
        return f"❌ Connection error: {e}"


def analyze_flight(metrics: dict, sampling_info: dict, filename: str,
                   api_key: Optional[str] = None) -> str:
    """
    Generate a Ukrainian-language AI analysis of the flight.

    Args:
        metrics:       Output of metrics.compute_metrics()
        sampling_info: Output of parser.get_sampling_info()
        filename:      Original log filename
        api_key:       Anthropic API key (optional, falls back to env var)

    Returns:
        Markdown-formatted analysis string.
    """
    prompt = f"""Ти — експерт з аналізу польотів безпілотних літальних апаратів (БПЛА).
Проаналізуй наступні метрики польоту та надай детальний висновок УКРАЇНСЬКОЮ МОВОЮ.

Файл логу: {filename}

=== МЕТРИКИ ПОЛЬОТУ ===
- Тривалість польоту: {metrics.get('duration_s', 0):.1f} с
- Загальна відстань (GPS Haversine): {metrics.get('total_distance_m', 0):.1f} м
- Макс. горизонтальна швидкість (GPS): {metrics.get('max_horiz_speed_ms', 0):.2f} м/с
- Макс. вертикальна швидкість (GPS VZ): {metrics.get('max_vert_speed_ms', 0):.2f} м/с
- Макс. прискорення (IMU): {metrics.get('max_accel_ms2', 0):.2f} м/с²
- Макс. набір висоти (відносно точки старту): {metrics.get('max_altitude_gain_m', 0):.1f} м
- Висота старту (MSL): {metrics.get('home_alt_m', 0):.1f} м
- Кількість GPS-фіксів: {metrics.get('gps_points', 0)}
- Кількість IMU-семплів: {metrics.get('imu_samples', 0)}

=== ЧАСТОТИ СЕМПЛЮВАННЯ ===
{json.dumps(sampling_info, ensure_ascii=False, indent=2)}

=== ЗАВДАННЯ ===
Надай структурований аналіз у форматі Markdown:

1. **Загальна оцінка місії** — чи виглядає політ нормальним, чи є аномалії?
2. **Аналіз швидкостей** — чи є різкі зміни, перевищення безпечних порогів?
3. **Аналіз висоти** — профіль набору/зниження, різкі зміни висоти?
4. **Аналіз прискорень** — чи є ознаки турбулентності, різких маневрів?
5. **Якість сигналу GPS** — оцінка за кількістю фіксів та тривалістю
6. **Рекомендації** — що варто перевірити або покращити?

Будь конкретним і технічним. Якщо значення виходять за межі норми — обов'язково вкажи це.
"""

    return _call_claude(prompt, api_key)