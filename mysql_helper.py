import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # Change as needed
    'password': 'students',  # Change as needed
    'database': 'physio_vr'
}

def get_mysql_connection():
    """
    Establishes and returns a reusable MySQL connection object.
    Returns:
        mysql.connector.connection.MySQLConnection: MySQL connection object
    Raises:
        mysql.connector.Error: If connection fails
    """
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        raise
