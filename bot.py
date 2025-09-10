from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re

# ---- Arithmetic functions ----
def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b): return "Error: Cannot divide by zero." if b == 0 else a / b

# ---- Training data ----
training_sentences = [
    "add 5 and 7", "sum of 4 and 5", "7 plus 8", "do addition", "perform addition",
    "subtract 10 and 4", "difference between 9 and 3", "take away 6 from 12",
    "do subtraction", "perform subtraction",
    "multiply 3 and 5", "product of 2 and 4", "7 times 8",
    "multiplication", "do multiplication", "perform multiplication",
    "divide 10 by 2", "quotient of 9 and 3", "split 12 into 4",
    "division", "do division", "perform division"
]

labels = [
    "add", "add", "add", "add", "add",
    "subtract", "subtract", "subtract", "subtract", "subtract",
    "multiply", "multiply", "multiply", "multiply", "multiply", "multiply",
    "divide", "divide", "divide", "divide", "divide", "divide"
]

# ---- Train ML model ----
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)
model = MultinomialNB()
model.fit(X, labels)

# ---- Process input ----
def process_input(user_input):
    user_input = user_input.lower()

    # Add spaces around operators
    for symbol in ["+", "-", "*", "/"]:
        user_input = user_input.replace(symbol, f" {symbol} ")

    # Handle simple arithmetic expressions like "5 + 7 + 3"
    if re.match(r'^([\d+\-*/.\s]+)$', user_input):
        try:
            allowed_chars = "0123456789+-*/. "
            clean_expr = ''.join([c for c in user_input if c in allowed_chars])
            return eval(clean_expr)
        except:
            return "Invalid expression"

    # Extract numbers from sentence
    numbers = re.findall(r'-?\d+\.?\d*', user_input)
    numbers = list(map(float, numbers))

    if len(numbers) < 2:
        return "Please provide at least two numbers."

    # Predict intent with ML model
    intent = model.predict(vectorizer.transform([user_input]))[0]

    # Perform operation iteratively on all numbers
    if intent == "add":
        result = numbers[0]
        for n in numbers[1:]:
            result = add(result, n)
        return result

    if intent == "subtract":
        result = numbers[0]
        for n in numbers[1:]:
            result = subtract(result, n)
        return result

    if intent == "multiply":
        result = numbers[0]
        for n in numbers[1:]:
            result = multiply(result, n)
        return result

    if intent == "divide":
        result = numbers[0]
        for n in numbers[1:]:
            result = divide(result, n)
            if isinstance(result, str):  # division by zero
                return result
        return result

    return "Sorry, I don't understand that."

# ---- FastAPI setup ----
app = FastAPI(title="Calculator Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    message: str

@app.post("/chat")
def chat(inp: ChatIn):
    print(f"User said: {inp.message}")
    result = process_input(inp.message)
    return {"result": str(result)}
