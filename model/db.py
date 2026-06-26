import mysql.connector

try:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="pass123",
        database="emotion_detection"
    )

    cursor = connection.cursor()

    print("MySQL Connected Successfully!")

except mysql.connector.Error as err:
    print("Database Connection Error:", err)
    exit()