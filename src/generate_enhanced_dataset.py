import pandas as pd
import random
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "fast_delivery_ml_project" / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "delivery_reviews_enhanced.csv"

# Realistic values for randomization (based on your CSV)
AGENT_NAMES = ["Zepto", "JioMart", "Blinkit", "Swiggy Instamart"]
LOCATIONS = ["Delhi", "Lucknow", "Ahmedabad", "Chennai", "Pune", "Mumbai", "Kolkata", "Hyderabad", "Bangalore", "Jaipur"]
ORDER_TYPES = ["Essentials", "Grocery", "Pharmacy", "Electronics", "Food"]
PRICE_RANGES = ["High", "Medium", "Low"]
DISCOUNT_APPLIED = ["Yes", "No"]
PRODUCT_AVAILABILITY = ["In Stock", "Out of Stock"]
ORDER_ACCURACY = ["Correct", "Incorrect"]

# Expanded review templates (50 per class)
NEGATIVE_REVIEWS = [
    "Delivery was over an hour late, items were incorrect.",
    "Wrong order received, customer service was unhelpful.",
    "Food arrived cold, packaging was damaged.",
    "Driver was rude and ignored delivery instructions.",
    "Items missing from my order, very disappointing.",
    "Took forever to deliver, no updates provided.",
    "Order cancelled without notice, wasted my time.",
    "Products out of stock, wasn’t informed earlier.",
    "Package left at wrong address, terrible service.",
    "Food spilled in the bag, completely unacceptable.",
    "Late delivery and poor communication throughout.",
    "Received damaged goods, no refund offered.",
    "Driver didn’t follow instructions, order was wrong.",
    "Service was slow, items were not as ordered.",
    "Delivery took too long, food was lukewarm.",
    "Missing items and no response from support.",
    "Package was mishandled, contents were broken.",
    "Order was delayed by hours, no apology given.",
    "Wrong items delivered, very frustrating experience.",
    "Customer service was unresponsive, bad experience.",
    "Delivery was late, and the app didn’t update.",
    "Items were out of stock, no prior warning.",
    "Driver was impolite, left package in wrong spot.",
    "Food arrived soggy, terrible quality control.",
    "Order was incomplete, support was unhelpful.",
    "Delivery was hours late, no tracking updates.",
    "Received wrong products, very poor service.",
    "Package was damaged, contents were unusable.",
    "Driver didn’t contact me, left order outside.",
    "Service was disorganized, delivery was late.",
    "Items were incorrect, no resolution offered.",
    "Food was cold and poorly packaged.",
    "Delivery took ages, no communication.",
    "Order was cancelled last minute, very annoying.",
    "Products were missing, support didn’t respond.",
    "Package was left in an unsafe location.",
    "Driver was rude, delivery was delayed.",
    "Items were damaged, no refund process.",
    "Order was wrong, customer service ignored me.",
    "Delivery was slow, tracking was inaccurate.",
    "Food arrived late and was not fresh.",
    "Missing items, no explanation provided.",
    "Package was mishandled, items broken.",
    "Driver ignored instructions, poor service.",
    "Order was delayed, no updates given.",
    "Wrong items delivered, very inconvenient.",
    "Service was slow, support was unhelpful.",
    "Delivery was late, food was cold.",
    "Items were out of stock, no notice given.",
    "Package was damaged, terrible experience."
]

NEUTRAL_REVIEWS = [
    "Delivery was on time, but tracking could improve.",
    "Order arrived fine, items were as expected.",
    "Service was okay, nothing special.",
    "Delivery was slightly delayed, driver was polite.",
    "Got my order, packaging could be better.",
    "Experience was average, needs more communication.",
    "Items delivered, one product was substituted.",
    "Service was decent, expected faster delivery.",
    "Order was correct, app didn’t update properly.",
    "Delivery was fine, customer service was slow.",
    "Order arrived on time, but not exceptional.",
    "Service was acceptable, could be smoother.",
    "Delivery was okay, tracking wasn’t great.",
    "Items were delivered, minor packaging issue.",
    "Order was fine, but app needs improvement.",
    "Delivery was timely, but communication lacked.",
    "Service was average, no major complaints.",
    "Order arrived, but tracking was unreliable.",
    "Delivery was okay, could use better updates.",
    "Items were correct, service was standard.",
    "Delivery was on time, app was glitchy.",
    "Order was fine, but driver seemed rushed.",
    "Service was decent, nothing stood out.",
    "Items arrived, packaging was adequate.",
    "Delivery was timely, but updates were slow.",
    "Order was correct, service was okay.",
    "Delivery was fine, but app could be better.",
    "Service was average, expected more.",
    "Items were delivered, minor delay occurred.",
    "Order arrived, tracking could be improved.",
    "Delivery was okay, communication was lacking.",
    "Service was standard, no issues but no wow factor.",
    "Order was fine, delivery was slightly late.",
    "Items arrived, but app didn’t update.",
    "Delivery was on time, service was average.",
    "Order was correct, but packaging was okay.",
    "Service was fine, could improve tracking.",
    "Items were delivered, no major problems.",
    "Delivery was okay, but updates were slow.",
    "Order arrived, service was standard.",
    "Delivery was timely, but app was slow.",
    "Items were correct, experience was average.",
    "Service was okay, tracking needs work.",
    "Order was fine, delivery was on time.",
    "Delivery was acceptable, but not great.",
    "Items arrived, service was decent.",
    "Order was correct, app could improve.",
    "Delivery was fine, communication was okay.",
    "Service was average, no complaints.",
    "Items were delivered, tracking was okay."
]

POSITIVE_REVIEWS = [
    "Super fast delivery, everything was perfect!",
    "Driver was friendly, followed instructions exactly.",
    "Order was accurate, arrived earlier than expected.",
    "Excellent service, products well-packaged.",
    "Great experience, will order again!",
    "Delivery was quick, food was hot.",
    "Perfect service, no issues, highly recommend.",
    "Items in stock, delivered on time, awesome!",
    "Professional driver, seamless delivery process.",
    "Fantastic experience, everything spot on!",
    "Delivery was lightning fast, great job!",
    "Order was perfect, driver was courteous.",
    "Service was outstanding, highly satisfied.",
    "Food arrived hot, packaging was excellent.",
    "Delivery was early, items were correct.",
    "Amazing service, very professional driver.",
    "Order was flawless, will use again!",
    "Fast delivery, everything was as ordered.",
    "Great customer service, quick delivery.",
    "Items were perfect, arrived right on time.",
    "Delivery was smooth, driver was friendly.",
    "Service was top-notch, no complaints!",
    "Order arrived early, packaging was great.",
    "Fantastic delivery, food was fresh.",
    "Driver was polite, order was accurate.",
    "Delivery was quick, service was excellent.",
    "Items were well-packaged, arrived on time.",
    "Perfect experience, highly recommend!",
    "Fast and reliable delivery, great service!",
    "Order was correct, driver was professional.",
    "Service was amazing, everything was perfect.",
    "Delivery was super fast, food was hot.",
    "Items arrived in great condition, on time.",
    "Driver was friendly, service was awesome.",
    "Order was perfect, delivery was quick.",
    "Great experience, everything went smoothly.",
    "Delivery was on time, items were correct.",
    "Service was excellent, driver was polite.",
    "Food arrived fresh, packaging was great.",
    "Delivery was fast, service was outstanding.",
    "Order was accurate, arrived early.",
    "Amazing delivery experience, no issues!",
    "Driver was professional, items were perfect.",
    "Service was great, delivery was quick.",
    "Order was spot on, arrived on time.",
    "Delivery was smooth, food was fresh.",
    "Great service, items were well-packaged.",
    "Fast delivery, driver was courteous.",
    "Perfect order, excellent customer service.",
    "Delivery was quick, everything was great!"
]

def vary_review(review):
    """Add slight variations to reviews for diversity"""
    words = review.split()
    if random.random() < 0.3:  # 30% chance to modify
        synonyms = {
            "fast": ["quick", "swift", "prompt"],
            "great": ["excellent", "awesome", "fantastic"],
            "late": ["delayed", "tardy", "slow"],
            "poor": ["bad", "terrible", "subpar"],
            "okay": ["fine", "decent", "acceptable"],
            "service": ["support", "assistance", "delivery"],
            "driver": ["courier", "delivery person", "rider"],
            "items": ["products", "goods", "order"]
        }
        for i, word in enumerate(words):
            for key, vals in synonyms.items():
                if word.lower() == key and random.random() < 0.5:
                    words[i] = random.choice(vals)
                    break
    return " ".join(words)

def generate_enhanced_dataset(n_negative=1727, n_neutral=1625, n_positive=1648):
    """Generate dataset matching original CSV structure and label distribution"""
    data = []
    
    for feedback_type, reviews, n_samples in [
        ("Negative", NEGATIVE_REVIEWS, n_negative),
        ("Neutral", NEUTRAL_REVIEWS, n_neutral),
        ("Positive", POSITIVE_REVIEWS, n_positive)
    ]:
        for _ in range(n_samples):
            review = vary_review(random.choice(reviews))  # Add variation
            row = {
                "Agent Name": random.choice(AGENT_NAMES),
                "Rating": round(random.uniform(1.0 if feedback_type == "Negative" else 3.0, 5.0 if feedback_type == "Positive" else 4.0), 1),
                "Review Text": review,
                "Delivery Time (min)": random.randint(30 if feedback_type == "Negative" else 20, 60 if feedback_type == "Negative" else 40),
                "Location": random.choice(LOCATIONS),
                "Order Type": random.choice(ORDER_TYPES),
                "Customer Feedback Type": feedback_type,
                "Price Range": random.choice(PRICE_RANGES),
                "Discount Applied": random.choice(DISCOUNT_APPLIED),
                "Product Availability": random.choice(PRODUCT_AVAILABILITY),
                "Customer Service Rating": random.randint(1 if feedback_type == "Negative" else 3, 5 if feedback_type == "Positive" else 4),
                "Order Accuracy": random.choice(ORDER_ACCURACY)
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Generating enhanced dataset with ~5000 samples...")
    df = generate_enhanced_dataset(n_negative=1727, n_neutral=1625, n_positive=1648)
    
    print(f"\nDataset Stats:")
    print(f"Total samples: {len(df)}")
    print(f"\nLabel distribution (Customer Feedback Type):")
    print(df["Customer Feedback Type"].value_counts())
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("Update DATA_FILE in training scripts to use this dataset.")