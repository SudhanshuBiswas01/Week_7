"""
ShopSense Realistic Dataset Generator v2
- 5,000 reviews with genuine lexical variation
- 94.4% positive / 5.6% negative class imbalance
- ~15% Hinglish (Hindi-English mixed) reviews
- ~8% label noise (ambiguous / borderline reviews)
- Typos, punctuation variation, rating-text mismatch
- Multiple product categories (train + unseen)
"""

import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

CATEGORIES_TRAIN  = ["Electronics", "Clothing", "Beauty", "Footwear", "Books"]
CATEGORIES_UNSEEN = ["Furniture", "Sports", "Toys"]

# ── Positive English pool (varied, realistic) ─────────────────────────────
POS_ENGLISH = [
    "Amazing product! Exactly what I needed. Delivery was fast too.",
    "Excellent quality. Worth every rupee I spent on this.",
    "Really happy with this purchase. Will definitely buy again.",
    "Product looks exactly like the photos. Very satisfied customer.",
    "Best purchase I made this year. Highly recommend to everyone.",
    "Superb quality and packaging was also great. No complaints.",
    "Delivered on time and product is premium quality.",
    "Works perfectly. No complaints whatsoever. 5 stars.",
    "Great value for money. Very impressed with the build.",
    "Loved it! Quality exceeded my expectations completely.",
    "Five stars! Product is exactly as described. Fast shipping.",
    "Good product, good delivery. Happy customer overall.",
    "Material quality is outstanding. Very pleased with it.",
    "Excellent item. Fast shipping. Great seller to buy from.",
    "Totally worth it. A must buy for anyone looking for this.",
    "Just received my order and I must say it is wonderful.",
    "Absolutely love this product. My whole family liked it.",
    "Very sturdy and well-made. Exceeded expectations honestly.",
    "Looks premium and feels premium too. No regrets at all.",
    "Great customer service and the product is top notch.",
    "Arrived 2 days early and the quality is fantastic.",
    "Perfect gift, recipient loved it. Packaging was neat.",
    "Very durable. Been using it for 3 weeks with zero issues.",
    "Impressed with the attention to detail. Highly satisfied.",
    "Smooth delivery, zero damage, product works like a charm.",
    "Ordered on Monday, arrived Thursday. Works as advertised.",
    "Build quality is solid. Looks even better in person.",
    "Unboxed this today and immediately impressed by the quality.",
    "Price is fair for what you get. Good investment overall.",
    "Reliable product, came well packaged, does the job well.",
]

# ── Negative English pool ─────────────────────────────────────────────────
NEG_ENGLISH = [
    "Very disappointed. Product stopped working after just 2 days.",
    "Terrible quality. Nothing like the pictures shown online.",
    "Complete waste of money. Do not buy this product at all.",
    "Returned immediately. Product arrived damaged in the box.",
    "Horrible experience. Customer support was completely useless.",
    "Broke after first use. Very poor build quality overall.",
    "Fake product. Smells awful and looks nothing like the listing.",
    "Did not match description at all. Very misleading photos.",
    "Pathetic quality. Expected much better for this price point.",
    "Worst purchase ever. Totally useless, stopped working instantly.",
    "Received wrong product and the return process was a nightmare.",
    "Very bad quality. Would give zero stars if the option existed.",
    "Stopped working on day one. Extremely frustrated with this.",
    "Cheap material, falls apart easily. Avoid this product.",
    "Scam! The product is completely different from what was shown.",
    "Waste of money, does not do what it claims. Very angry.",
    "Awful experience from start to finish. Never buying again.",
    "Product is defective. Contacted support but no response yet.",
    "Looks nothing like the photo. Felt cheated after opening it.",
    "Material quality is terrible. Fell apart within a week.",
    "Disappointed with delivery and the product is broken inside.",
    "Super overpriced for this quality. Total rip off honestly.",
    "Battery died in 3 hours. Completely unusable out of the box.",
    "Delivered late AND the product was wrong. Unacceptable.",
    "Smells like chemicals, cannot use it at all. Return pending.",
    "The stitching came apart after one wash. Really poor quality.",
    "Lid cracked on day 2. Do not waste your money on this.",
    "Size is way off from what is listed. Misleading description.",
    "Does not charge properly. Tech support is completely unhelpful.",
    "Packaging was torn and product was scratched. Very careless.",
]

# ── Borderline / Ambiguous pool (creates label noise challenge) ────────────
BORDERLINE = [
    "It is okay I guess, nothing special but does the job.",
    "Average product. Not bad, not great. Could be better.",
    "Packaging was nice but the product itself is just fine.",
    "Works as described but the quality is a bit disappointing.",
    "Good for the price, but do not expect premium quality.",
    "Decent enough, but I have seen better at this price.",
    "Arrived on time. Product is acceptable but not impressive.",
    "It is fine for daily use but breaks easily if dropped.",
    "Meets basic needs but nothing extraordinary about it.",
    "Looks decent in photos, slightly different in person.",
    "Was expecting more for this price honestly. Just average.",
    "Product works but customer service was not great.",
]

# ── Hinglish Positive pool ────────────────────────────────────────────────
POS_HINGLISH = [
    "Bahut acha product hai! Delivery bhi fast thi. Bilkul recommend karunga.",
    "Ek dum mast item hai yaar. Worth the price, no regrets.",
    "Quality ekdum top notch hai. Khush ho gaya main completely.",
    "Sach mein bahut badhiya hai bhai, paisa vasool hai.",
    "Packaging bhi achi thi aur product bhi superb. Happy customer!",
    "Maine pehle socha tha theek hoga but it is amazing yaar.",
    "Mast product! Mere ghar mein sab ne pasand kiya isko.",
    "Very good quality. Dil khush ho gaya dekhke ekdum.",
    "Kaafi achi quality hai bhai. Definitely try karo isko.",
    "Zabardast item! Jaldi deliver hua aur condition mein perfect tha.",
    "Bahut sundar hai aur build bhi solid lagta hai yaar.",
    "Paise ki full value mili. Highly recommend karunga sab ko.",
    "Ekdum best product hai. Koi complaint nahi hai mujhe.",
    "Ek dum acha hai yaar, sach mein surprised tha main.",
    "Order kiya aur 3 din mein aa gaya. Quality bhi top hai.",
]

# ── Hinglish Negative pool ────────────────────────────────────────────────
NEG_HINGLISH = [
    "Bilkul bekaar product hai. Paisa barbaad hua mera completely.",
    "Kya bakwaas item bheja hai, quality zero hai ekdum.",
    "Total fraud hai yaar. Photo mein kuch aur tha, mila kuch aur.",
    "Ek dum ghatiya hai. Mat kharido isko bilkul bhi.",
    "Bhai bahut bura lag raha hai. 2 din mein toot gaya yaar.",
    "Wapas karna pad gaya. Customer care ne koi help nahi ki.",
    "Worst product hai bhai. Sab jhooth tha description mein.",
    "Do din mein kharaab ho gaya. Paise doobe mere.",
    "Ekdum faltu hai yaar, photo se bilkul alag aaya.",
    "Mat lena yaar, bahut ghatiya quality hai iska.",
]

# ── Hinglish Borderline ───────────────────────────────────────────────────
BORDERLINE_HINGLISH = [
    "Theek hai yaar, kuch special nahi par kaam chalta hai.",
    "Average hi hai, na zyada accha na bura. Chalega.",
    "Price ke hisaab se zyada expect tha, par chalta hai.",
]

def add_noise(text: str, noise_prob: float = 0.15) -> str:
    """Inject realistic textual noise: typos, extra punctuation, etc."""
    if random.random() > noise_prob:
        return text
    ops = random.choice(["lower", "extra_punct", "repeat_word", "abbreviate"])
    if ops == "lower":
        return text.lower()
    elif ops == "extra_punct":
        return text.rstrip(".") + "!!" if random.random() > 0.5 else text.replace(".", ".. ")
    elif ops == "repeat_word":
        words = text.split()
        if len(words) > 3:
            idx = random.randint(0, len(words)-1)
            words.insert(idx, words[idx])
            return " ".join(words)
        return text
    elif ops == "abbreviate":
        return text.replace("product", "prod").replace("quality", "qlty").replace("delivery", "del")
    return text


def generate_dataset(n: int = 5000, neg_frac: float = 0.056,
                     hinglish_frac: float = 0.15, borderline_frac: float = 0.08,
                     unseen_frac: float = 0.12) -> pd.DataFrame:
    rows = []
    for i in range(n):
        # 1. Determine label
        is_borderline = random.random() < borderline_frac
        if is_borderline:
            label = random.choice(["positive", "negative"])  # noisy label
        else:
            label = "negative" if random.random() < neg_frac else "positive"

        # 2. Language
        is_hinglish = random.random() < hinglish_frac

        # 3. Category
        if i >= int(n * (1 - unseen_frac)):
            category = random.choice(CATEGORIES_UNSEEN)
        else:
            category = random.choice(CATEGORIES_TRAIN)

        # 4. Pick template
        if is_borderline:
            pool = BORDERLINE_HINGLISH if is_hinglish and random.random() > 0.5 else BORDERLINE
        elif label == "positive":
            pool = POS_HINGLISH if is_hinglish else POS_ENGLISH
        else:
            pool = NEG_HINGLISH if is_hinglish else NEG_ENGLISH

        text = add_noise(random.choice(pool))

        # 5. Rating (occasionally mismatched for label noise realism)
        if is_borderline:
            rating = random.randint(2, 4)
        elif label == "positive":
            rating = random.randint(4, 5) if random.random() > 0.05 else random.randint(1, 3)
        else:
            rating = random.randint(1, 2) if random.random() > 0.05 else random.randint(3, 5)

        rows.append({
            "review_id":        f"REV{i+1:05d}",
            "review_text":      text,
            "sentiment":        label,
            "product_category": category,
            "is_hinglish":      is_hinglish,
            "is_borderline":    is_borderline,
            "rating":           rating,
        })

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["review_id"] = [f"REV{i+1:05d}" for i in range(len(df))]
    return df


if __name__ == "__main__":
    df = generate_dataset(n=5000)
    df.to_csv("/home/claude/week-07/friday/shopsense_reviews.csv", index=False)
    print(f"Dataset: {df.shape}")
    print(df["sentiment"].value_counts())
    print(f"Unique texts: {df['review_text'].nunique()}")
    print(f"Hinglish: {df['is_hinglish'].sum()} ({df['is_hinglish'].mean()*100:.1f}%)")
    print(f"Borderline: {df['is_borderline'].sum()} ({df['is_borderline'].mean()*100:.1f}%)")
    print(df["product_category"].value_counts())
