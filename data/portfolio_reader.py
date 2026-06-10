"""
Uses Claude vision to extract portfolio positions from a screenshot.
Returns a list of {ticker, quantity, avg_price, current_value}.
"""
import base64
import json
import re
import anthropic


def image_to_base64(image_bytes: bytes) -> str:
    return base64.standard_b64encode(image_bytes).decode("utf-8")


def extract_portfolio_from_image(image_bytes: bytes, media_type: str = "image/png") -> list[dict]:
    client = anthropic.Anthropic()

    prompt = """Tu regardes un screenshot de portefeuille boursier.
Extrait TOUTES les positions visibles et retourne UNIQUEMENT un JSON valide, sans texte autour.

Format attendu (tableau JSON) :
[
  {
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "quantity": 10.5,
    "avg_price": 150.00,
    "current_price": 175.00,
    "current_value": 1837.50,
    "gain_loss_pct": 16.67
  }
]

Règles :
- ticker : symbole boursier officiel (ex: AAPL, MSFT, SPY). Si tu vois un nom d'entreprise mais pas le ticker, déduis-le.
- quantity : nombre d'actions (float, peut être null si non visible)
- avg_price : prix moyen d'achat (float, peut être null si non visible)
- current_price : prix actuel (float, peut être null si non visible)
- current_value : valeur totale de la position (float, peut être null si non visible)
- gain_loss_pct : performance en % (float, peut être null si non visible)

Si une valeur n'est pas lisible, mets null.
Retourne UNIQUEMENT le JSON, rien d'autre."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_to_base64(image_bytes),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    return json.loads(raw)


def generate_portfolio_recommendations(positions: list[dict], analyses: dict) -> str:
    """
    Given extracted positions + per-ticker analyses, ask Claude to produce
    a structured rebalancing recommendation.
    """
    client = anthropic.Anthropic()

    summary_lines = []
    for pos in positions:
        t = pos["ticker"]
        a = analyses.get(t, {})
        sc = a.get("scenarios", {})
        dominant = a.get("dominant", "N/A")
        bull_p = sc.get("bull", {}).get("probability", "?")
        base_p = sc.get("base", {}).get("probability", "?")
        bear_p = sc.get("bear", {}).get("probability", "?")
        fund = a.get("fund", {})
        rec = fund.get("recommendation", "N/A")
        target = fund.get("analyst_target")
        current = pos.get("current_price") or pos.get("current_value")
        gain = pos.get("gain_loss_pct")

        line = (
            f"- {t}: qty={pos.get('quantity')}, "
            f"prix moyen={pos.get('avg_price')}, "
            f"prix actuel={pos.get('current_price')}, "
            f"P&L={gain}%, "
            f"scénario dominant={dominant} (Bull {bull_p}% / Base {base_p}% / Bear {bear_p}%), "
            f"consensus analystes={rec}, objectif={target}"
        )
        summary_lines.append(line)

    portfolio_text = "\n".join(summary_lines)

    prompt = f"""Tu es un conseiller en gestion de portefeuille. Voici l'état actuel d'un portefeuille avec les analyses techniques et fondamentales de chaque position :

{portfolio_text}

Produis une analyse de portefeuille structurée en français avec :

1. **Résumé global** — évaluation de l'équilibre du portefeuille (diversification, exposition au risque, biais sectoriels)

2. **Analyse position par position** — pour chaque ticker :
   - Statut : ✅ Conserver / ⚠️ Surveiller / 🔴 Alléger / 💡 Renforcer
   - Justification courte (2-3 lignes)
   - Action suggérée concrète

3. **Propositions d'ajustement** — 3 à 5 recommandations prioritaires avec :
   - L'action exacte (ex: "Alléger NVDA de 30%", "Initier une position sur QQQ")
   - La raison
   - Le niveau de priorité (Urgent / Important / Optionnel)

4. **Scénario global du portefeuille** — probabilité Bull/Base/Bear pour l'ensemble du portefeuille avec une explication

Sois précis, factuel, et base-toi uniquement sur les données fournies."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text
