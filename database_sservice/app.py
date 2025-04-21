from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# Создание базы данных с таблицами users и products
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        fio TEXT,
        nickname TEXT,
        balance INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category INTEGER,
        name TEXT,
        price REAL,
        quantity INTEGER
    )
    """)

    conn.commit()
    conn.close()

@app.route('/add_product', methods=['POST'])
def add_product():
    data = request.json
    category = data.get("category")
    name = data.get("name")
    price = data.get("price")
    quantity = data.get("quantity")

    if None in (category, name, price, quantity):
        return jsonify({"error": "Missing fields"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO products (category, name, price, quantity) VALUES (?, ?, ?, ?)", 
                   (category, name, price, quantity))
    conn.commit()
    conn.close()

    return jsonify({"message": "Product added successfully"}), 201

@app.route('/get_products', methods=['GET'])
def get_products():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products")
    products = cursor.fetchall()
    conn.close()

    return jsonify([dict(product) for product in products])

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5003, debug=True)
