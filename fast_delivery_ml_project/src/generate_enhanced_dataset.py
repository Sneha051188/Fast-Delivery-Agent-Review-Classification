"""
Enhanced Dataset Generator for Fast Delivery Agent Review Classification
Generates 2000 balanced samples (667 Negative, 667 Neutral, 666 Positive)
"""
import pandas as pd
import random
import os
from pathlib import Path

# Define review templates for each sentiment class
NEGATIVE_REVIEWS = [
    "Terrible delivery experience. Driver was rude and unprofessional.",
    "My order arrived completely damaged. Very disappointed with the service.",
    "Worst delivery service ever. Never ordering again.",
    "Driver didn't follow delivery instructions at all. Food was cold.",
    "Absolutely horrible service. Waited 2 hours for nothing.",
    "The delivery person was extremely rude and threw my package.",
    "Food arrived spoiled and the driver refused to apologize.",
    "Never received my order. Customer service was no help.",
    "Packaging was torn and items were missing from my order.",
    "Driver left my order in the rain. Everything was ruined.",
    "Unprofessional behavior from start to finish. Very unhappy.",
    "Order was wrong and driver argued with me about it.",
    "Late delivery, cold food, and rude driver. Complete disaster.",
    "Will never use this service again. Absolutely terrible.",
    "Driver didn't wear mask and was coughing. Very unsafe.",
    "My food was delivered to wrong address. Had to search for it.",
    "Extremely poor quality service. Driver didn't care at all.",
    "Order arrived 3 hours late and was completely cold.",
    "Driver was on phone entire time. Very unprofessional.",
    "Worst experience ever. Food was spilled all over the bag.",
    "Driver demanded extra tip aggressively. Very uncomfortable.",
    "Package was clearly tampered with. Items were missing.",
    "Horrible communication. Driver never responded to calls.",
    "Food quality was terrible. Clearly not handled properly.",
    "Driver parked in wrong spot and made me walk far to get order.",
    "Very disappointed. Order was incomplete and cold.",
    "Terrible service from beginning to end. Never again.",
    "Driver was lost for 45 minutes. Food arrived cold and soggy.",
    "Unprofessional and careless. My order was thrown at door.",
    "Worst delivery I've ever experienced. Complete waste of money.",
    "Driver left order outside gate in the sun. Food was hot and spoiled.",
    "Absolutely unacceptable service. Driver was very rude.",
    "My special instructions were completely ignored.",
    "Food arrived spilled everywhere. Driver didn't apologize.",
    "Terrible experience. Will be filing a complaint.",
    "Driver was smoking while handling my food. Disgusting.",
    "Order was delayed for hours with no communication.",
    "Very poor service. Driver had bad attitude throughout.",
    "Food was cold, late, and wrong. Triple disappointment.",
    "Driver didn't bring utensils despite request. Very careless.",
    "Horrible experience from start to finish.",
    "My order was left on ground in dirt. Very unsanitary.",
    "Driver was texting while driving. Very unsafe behavior.",
    "Terrible service. Order was completely wrong.",
    "Food arrived crushed. Packaging was inadequate.",
    "Driver was unprofessional and argumentative.",
    "Worst delivery service in the area. Avoid at all costs.",
    "My food was clearly old and reheated. Unacceptable.",
    "Driver left order at wrong house. Had to search neighborhood.",
    "Absolutely horrible. Food was cold and driver was rude."
]

NEUTRAL_REVIEWS = [
    "Delivery was okay. Nothing special but nothing terrible either.",
    "Average service. Order arrived on time but food was lukewarm.",
    "Decent delivery. Could be better but wasn't bad.",
    "Normal delivery experience. Nothing to complain about.",
    "Service was acceptable. Met basic expectations.",
    "Standard delivery. Nothing stood out as good or bad.",
    "Fair service overall. Room for improvement.",
    "Delivery was fine. About what I expected.",
    "Okay experience. Nothing memorable either way.",
    "Average delivery time. Food temperature was okay.",
    "Service was satisfactory. Nothing more, nothing less.",
    "Decent enough. Would order again if needed.",
    "Standard service. No complaints but no praise either.",
    "Delivery was alright. Could have been better.",
    "Fair experience overall. Nothing special.",
    "Service met basic requirements. That's about it.",
    "Okay delivery. Food arrived in acceptable condition.",
    "Average experience. Nothing to write home about.",
    "Delivery was fine, though driver seemed rushed.",
    "Acceptable service. Nothing particularly good or bad.",
    "Standard delivery experience. As expected.",
    "Service was okay. Driver was polite but not friendly.",
    "Fair delivery. Food was slightly warm.",
    "Decent enough service for the price.",
    "Nothing special about this delivery.",
    "Average experience overall. Would order again.",
    "Service was acceptable but not impressive.",
    "Delivery was fine. No major issues.",
    "Okay experience. Driver was professional enough.",
    "Standard service. Order arrived as scheduled.",
    "Fair delivery experience. Nothing notable.",
    "Service was adequate. Met minimum expectations.",
    "Average delivery. Food quality was okay.",
    "Decent service. Could be improved slightly.",
    "Normal experience. Nothing out of ordinary.",
    "Service was satisfactory overall.",
    "Fair enough. Order was complete and on time.",
    "Average delivery experience. Would use again.",
    "Okay service. Driver followed basic instructions.",
    "Standard delivery. No surprises good or bad.",
    "Acceptable experience. Nothing to complain about.",
    "Service was fine. Food arrived in okay condition.",
    "Decent delivery. Driver was courteous enough.",
    "Fair experience. Met basic expectations.",
    "Average service overall. Nothing special.",
    "Delivery was okay. Would order again if convenient.",
    "Standard experience. As expected for the service.",
    "Service was acceptable. Driver was polite.",
    "Fair delivery. Food temperature was adequate.",
    "Okay experience overall. Nothing remarkable."
]

POSITIVE_REVIEWS = [
    "Excellent service! Driver was very professional and friendly.",
    "Amazing delivery experience. Food arrived hot and fresh!",
    "Best delivery service in town. Highly recommend!",
    "Driver was courteous and followed all instructions perfectly.",
    "Outstanding service. Order arrived early and in perfect condition.",
    "Very impressed with the professionalism. Will definitely order again!",
    "Fantastic delivery! Driver was polite and careful with my order.",
    "Great experience from start to finish. Five stars!",
    "Wonderful service. Driver went above and beyond!",
    "Excellent delivery time. Food was hot and delicious.",
    "Very pleased with the service. Driver was friendly and professional.",
    "Perfect delivery! Everything was exactly as ordered.",
    "Amazing driver! Very courteous and followed instructions perfectly.",
    "Outstanding service. Best delivery experience I've had.",
    "Highly recommend this service. Driver was excellent!",
    "Great delivery! Food arrived quickly and in perfect condition.",
    "Wonderful experience. Driver was polite and professional.",
    "Excellent service! Will definitely use again.",
    "Very impressed! Driver was friendly and efficient.",
    "Perfect delivery experience. Everything was great!",
    "Outstanding service from beginning to end.",
    "Amazing driver! Very professional and careful.",
    "Great service! Order arrived hot and fresh.",
    "Wonderful delivery! Driver was courteous and friendly.",
    "Excellent experience! Highly satisfied with the service.",
    "Very happy with the delivery. Driver was professional.",
    "Perfect service! Everything arrived in excellent condition.",
    "Outstanding delivery! Will definitely order again.",
    "Amazing experience! Driver was very polite and careful.",
    "Great job! Food arrived quickly and was still hot.",
    "Wonderful service! Driver followed all instructions.",
    "Excellent delivery experience. Very satisfied!",
    "Very impressed with the professionalism and speed.",
    "Perfect delivery! Driver was friendly and efficient.",
    "Outstanding service! Everything was perfect.",
    "Amazing delivery! Food arrived in excellent condition.",
    "Great experience! Will definitely recommend to friends.",
    "Wonderful driver! Very professional and courteous.",
    "Excellent service from start to finish!",
    "Very pleased! Delivery was quick and professional.",
    "Perfect experience! Driver was excellent.",
    "Outstanding delivery! Everything exceeded expectations.",
    "Amazing service! Driver was very friendly.",
    "Great delivery! Food was hot and packaging was perfect.",
    "Wonderful experience! Highly recommend this service.",
    "Excellent driver! Very polite and professional.",
    "Very satisfied! Delivery was quick and efficient.",
    "Perfect service! Will definitely use again soon.",
    "Outstanding experience! Driver was amazing.",
    "Great service! Everything was exactly as expected."
]

def generate_enhanced_dataset():
    """Generate balanced dataset with 2000 samples"""
    reviews = []
    
    # Generate 667 negative reviews
    for _ in range(667):
        review = random.choice(NEGATIVE_REVIEWS)
        reviews.append({"review": review, "sentiment": "Incorrect"})
    
    # Generate 667 neutral reviews
    for _ in range(667):
        review = random.choice(NEUTRAL_REVIEWS)
        reviews.append({"review": review, "sentiment": "Neutral"})
    
    # Generate 666 positive reviews
    for _ in range(666):
        review = random.choice(POSITIVE_REVIEWS)
        reviews.append({"review": review, "sentiment": "Correct"})
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(reviews)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_path = output_dir / "delivery_reviews_enhanced.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated enhanced dataset with {len(df)} samples")
    print(f"📊 Distribution:")
    print(df['sentiment'].value_counts())
    print(f"💾 Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    generate_enhanced_dataset()
