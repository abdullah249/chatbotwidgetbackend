import json
from groq import Groq

GROQ_API_KEY = "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"

groq_client = Groq(api_key=GROQ_API_KEY)

def handler(request):
    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Method not allowed"})
        }

    try:
        body = request.json()
        message = body.get("message", "")

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions clearly and concisely."
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=1024,
        )

        response = chat_completion.choices[0].message.content

        return {
            "statusCode": 200,
            "body": json.dumps({"response": response})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"response": f"Error: {str(e)}"})
        } 