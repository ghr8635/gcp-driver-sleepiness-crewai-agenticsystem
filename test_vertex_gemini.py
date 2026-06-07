import os
from google import genai

project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

print("Project:", project)
print("Location:", location)

if not project:
    raise ValueError("Project ID is missing. Set GOOGLE_CLOUD_PROJECT or PROJECT_ID.")

client = genai.Client(
    vertexai=True,
    project=project,
    location=location,
)

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=(
        "Return exactly these four lines only:\n"
        "Fan: Level 2\n"
        "Music: On\n"
        "Vibration: Off\n"
        "Reason: test"
    ),
)

print(response.text)