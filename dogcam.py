import cv2
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account
from dotenv import load_dotenv, find_dotenv
from os import getenv
from time import sleep, mktime
from datetime import datetime

# Load our credentials from a .env file
load_dotenv(find_dotenv())
google_creds = {
  "type": "service_account",
  "project_id": getenv("GOOGLE_PROJECT_ID"),
  "private_key_id": getenv("GOOGLE_PRIVATE_KEY_ID"),
  "private_key": getenv("GOOGLE_PRIVATE_KEY").replace('\\\\n', '\n'),
  "client_email": getenv("GOOGLE_CLIENT_EMAIL"),
  "client_id": getenv("GOOGLE_CLIENT_ID"),
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://accounts.google.com/o/oauth2/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": getenv("GOOGLE_CLIENT_X509_CERT_URL")
}
def dispense_dog_food():
  pass

# Create the Google Comp Vision client
credentials = service_account.Credentials.from_service_account_info(google_creds)
client = vision.ImageAnnotatorClient(credentials=credentials)

# Open video capture and define last feed variable
cap = cv2.VideoCapture(0)
last_feed = None

while (cap.isOpened()):
  print("Scanning the room for dog...")
  ret, frame = cap.read()
  # Show frame we captured without blocking the thread
  cv2.imshow('frame', frame)
  cv2.waitKey(1)
  # Send the image up to Google Cloud Vision
  image = cv2.imencode('.jpg', frame)[1].tostring()
  response = client.label_detection(image=types.Image(content=image))
  # Look through every label; print it out and feed the dog if needed
  for label in response.label_annotations:
    desc, score = label.description, label.score
    print("I found a '{}' with confidence of {}".format(desc, score))
    dog_detected = desc == "dog" and score > 0.75
    if dog_detected:
      print("Dog is in sight!")
      now = mktime(datetime.now().timetuple())
      # Only feed the dog every 6 hours so he is not obese
      hrs_last_fed = float(now - last_feed) / 60 / 60 \
                              if last_feed is not None else 6.0
      if hrs_last_fed >= 6.0:
        print("Dog is hungry. Feeding...")
        last_feed = mktime(datetime.now().timetuple())
        dispense_dog_food()
      else:
        print("Dog was fed only {} hours ago. Ignoring...".format(hrs_last_fed))
      break
  else:
    print("No dog detected here...")
  # Re-capture every 5 seconds so as to not spam Google
  sleep(5)

cap.release()
