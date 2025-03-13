from linebot.models import FlexSendMessage


from linebot.models import FlexSendMessage

from linebot.models import FlexSendMessage

def generate_flex_message(food_name, analysis_data, nutrition_data):
    """
    產生 LINE Flex Message，將分析結果分成 優點 / 潛在風險 / 建議
    """
    def format_section(title, items, color):
        """ 格式化優點、風險、建議區塊 """
        return [
            {"type": "text", "text": title, "weight": "bold", "size": "md", "color": color},
            {"type": "text", "text": "\n".join(["• " + item for item in items]), "wrap": True, "size": "sm", "color": "#666666"} if items else {},
            {"type": "separator", "margin": "md"}
        ]

    flex_message = {
        "type": "bubble",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": food_name,
                    "weight": "bold",
                    "size": "xl",
                    "color": "#1DB446",
                    "align": "start"
                }
            ]
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": (
                format_section("📌 優點", analysis_data.get("優點", []), "#1E90FF") +
                format_section("⚠️ 潛在風險", analysis_data.get("潛在風險", []), "#FF4500") +
                format_section("✅ 建議", analysis_data.get("建議", []), "#008000") +
                [
                    {"type": "text", "text": "🔥 熱量", "weight": "bold", "size": "lg", "color": "#FF4500"},
                    {"type": "text", "text": f"{nutrition_data.get('calories', 'N/A')} kcal", "weight": "bold", "size": "xxl", "align": "center", "color": "#FF4500"},
                    {"type": "separator", "margin": "md"},
                    {"type": "box", "layout": "vertical", "spacing": "sm", "contents": [
                        {"type": "box", "layout": "horizontal", "contents": [
                            {"type": "text", "text": "🍞 碳水化合物", "size": "sm", "color": "#555555"},
                            {"type": "text", "text": f"{nutrition_data.get('carbohydrate', 'N/A')} g", "size": "sm", "align": "end", "color": "#111111"}
                        ]},
                        {"type": "box", "layout": "horizontal", "contents": [
                            {"type": "text", "text": "🍬 糖分", "size": "sm", "color": "#555555"},
                            {"type": "text", "text": f"{nutrition_data.get('sugar', 'N/A')} g", "size": "sm", "align": "end", "color": "#111111"}
                        ]},
                        {"type": "box", "layout": "horizontal", "contents": [
                            {"type": "text", "text": "🍗 蛋白質", "size": "sm", "color": "#555555"},
                            {"type": "text", "text": f"{nutrition_data.get('protein', 'N/A')} g", "size": "sm", "align": "end", "color": "#111111"}
                        ]},
                        {"type": "box", "layout": "horizontal", "contents": [
                            {"type": "text", "text": "🥑 脂肪", "size": "sm", "color": "#555555"},
                            {"type": "text", "text": f"{nutrition_data.get('fat', 'N/A')} g", "size": "sm", "align": "end", "color": "#111111"}
                        ]}
                    ]}
                ]
            )
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#1DB446",
                    "action": {
                        "type": "postback",
                        "label": "🔍 查看完整營養資訊",
                        "data": f"full_nutrition:{food_name}"
                    }
                }
            ]
        }
    }

    return FlexSendMessage(alt_text=f"{food_name} 營養資訊", contents=flex_message)


