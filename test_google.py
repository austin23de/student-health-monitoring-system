import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

credentials = ServiceAccountCredentials.from_json_keyfile_name(
    "credentials.json",
    scope
)

client = gspread.authorize(credentials)

sheet = client.open(
    "Student Health Monitoring System"
).worksheet("Live_Records")

sheet.append_row([
    "ST001",
    "Frank Lee",
    36.7,
    78,
    "120/80",
    98,
    24.5,
    0,
    "NORMAL",
    "OK",
    "2025-08-01 10:00:00"
])

print("CONNECTED SUCCESSFULLY")
