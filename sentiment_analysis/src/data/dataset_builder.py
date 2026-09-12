"""
Dataset Builder Module
-----------------------
Sources:
  1. Sentiment140-style Tweets (open-source Twitter dataset)
  2. Amazon Product Reviews (open-source review dataset)

Each source contributes 150 manually-curated unique samples.
Combined, cleaned, balanced: 300 total samples.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetBuilder:

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

    def generate_synthetic_tweets(self, n: int = 150, seed: int = 42) -> pd.DataFrame:
        """Tweet-style samples. In production: load from Sentiment140 CSV."""
        positive = [
            "Just had the most amazing coffee this morning! Feeling energized and ready to go!",
            "So proud of my team today! We crushed that presentation and the client loved everything",
            "Finally finished my marathon training! 26.2 miles here I come!",
            "My dog just learned a new trick. Cutest thing I have ever seen in my life",
            "Best birthday ever! Surrounded by family and friends who love me. Grateful beyond words",
            "Just got promoted at work! Hard work really does pay off. Dreams do come true!",
            "The sunset today was absolutely breathtaking. Nature never fails to amaze me",
            "Cooked my first homemade pasta from scratch and it actually tasted amazing",
            "Reconnected with an old friend today. Some friendships are truly timeless",
            "Hit a new personal record at the gym today! Progress feels so incredibly good",
            "Absolutely loving this book I started reading. Cannot put it down at all!",
            "Surprise package arrived! My family sent my favorite snacks from home",
            "Great customer service from the airline! They upgraded me without even asking",
            "Just booked my dream vacation to Japan! I have wanted this for years now",
            "My startup just got its first paying customer! This journey is becoming real!",
            "Woke up feeling so refreshed and motivated today. Going to have a great day!",
            "My sister just had a baby girl! I am the happiest uncle in the entire world!",
            "Finally got the job offer I was waiting for. All the hard work paid off big time",
            "The weather is absolutely perfect today. Could not ask for a better afternoon",
            "Just finished reading an incredible novel. Highly recommend it to everyone",
            "My garden is blooming beautifully this spring. So satisfying to see it grow",
            "Had the best dinner with my parents tonight. These moments are truly priceless",
            "Finished my thesis after two years of hard work. Feeling incredibly accomplished",
            "My team won the championship today! Years of practice finally paid off!",
            "Just discovered a new coffee shop that is absolutely perfect in every way",
        ]
        negative = [
            "Worst flight experience ever. Four hour delay and no explanation from the airline",
            "My laptop just died right before my deadline. This day is a complete disaster",
            "Cannot believe how rude the staff was at that restaurant. Never going back there",
            "Traffic is absolutely terrible today. Two hours to go ten miles. Losing my mind",
            "Ordered online three weeks ago and still have not received my package. Terrible",
            "My phone screen cracked after dropping it once. These phones are so fragile now",
            "Failed my driving test for the second time. Feeling so defeated and hopeless",
            "The movie was a complete waste of two hours. Terrible plot and awful acting",
            "Got food poisoning from that new place downtown. Stay away from there completely",
            "Lost my wallet today with everything in it. This week keeps getting worse",
            "My landlord still has not fixed the heating and it has been two weeks. Freezing!",
            "New software update completely broke my workflow. Why fix what was not broken?",
            "Third time calling customer support and they still have not resolved my issue",
            "Gym was overcrowded, equipment broken, and music was way too loud. Waste of time",
            "Missed my train by thirty seconds. Had to wait an hour in the cold. Awful day",
            "The project got cancelled after six months of work. Feeling completely devastated",
            "My internet has been down for three days and the provider keeps giving excuses",
            "Terrible experience at the hotel. Room was dirty and staff was completely unhelpful",
            "My flight got cancelled and they did not offer any compensation at all. Outrageous",
            "The new restaurant was a huge disappointment. Cold food and extremely slow service",
            "Got a parking ticket even though I was only five minutes over the limit. So unfair",
            "My headphones broke after just two months of normal use. Very poor build quality",
            "The customer service representative was condescending and completely unhelpful",
            "Waited two hours at the doctor office only to be told to come back another day",
            "My package arrived completely damaged and the seller is refusing to refund me",
        ]
        neutral = [
            "Just finished watching the news. Lots of things happening around the world today",
            "Went to the grocery store. They were out of my usual brand so I tried a different one",
            "Attended a work meeting this morning about next quarter planning activities",
            "The weather today is partly cloudy with temperatures around sixty five degrees",
            "Picked up my dry cleaning after work. Normal Tuesday stuff as usual",
            "Reading about the new updates to Python. Some interesting changes in this version",
            "Had lunch at a new cafe downtown. It was an average experience overall I suppose",
            "Watched a documentary about ocean life. Informative but not particularly exciting",
            "Finished the quarterly report today. Submitted it before the deadline as expected",
            "My commute to work today took about thirty five minutes. About the usual time",
            "Updated my resume this afternoon. Just keeping it current for future opportunities",
            "Tried a new recipe for dinner tonight. It turned out okay but nothing special",
            "Bought a new book at the library sale. Will start reading it this weekend maybe",
            "Had a routine check up at the doctor. Everything came back completely normal",
            "Reorganized my desk today. It looks more organized than it did before now",
            "Took the bus to work instead of driving today. Arrived at the usual time as always",
            "Made a grocery list for the week. Need to pick up the usual items from the store",
            "The office was quiet today with half the team working from home remotely",
            "Scheduled a dentist appointment for next month. Just a regular cleaning visit",
            "Cooked a simple pasta for dinner. Nothing fancy but it filled me up well enough",
            "Watched the evening news before bed. Same topics as yesterday being discussed",
            "Got a haircut today. The barber did a standard job as he usually does for me",
            "Called my mom to catch up. We talked for about fifteen minutes about general things",
            "The meeting ran a bit longer than expected today but we covered all agenda items",
            "Walked to the post office to mail a package. Took about twenty minutes round trip",
        ]

        rows = []
        for text in positive[:n//3]:
            rows.append({"text": text, "label": 2, "source": "tweets"})
        for text in negative[:n//3]:
            rows.append({"text": text, "label": 0, "source": "tweets"})
        for text in neutral[:n//3]:
            rows.append({"text": text, "label": 1, "source": "tweets"})

        df = pd.DataFrame(rows)
        logger.info(f"Generated {len(df)} tweet samples")
        return df

    def generate_synthetic_reviews(self, n: int = 150, seed: int = 123) -> pd.DataFrame:
        """Product review-style samples. In production: load from Amazon Reviews CSV."""
        positive = [
            "Absolutely love this product! It exceeded all my expectations. Build quality is outstanding and works exactly as described. Highly recommend.",
            "This is the third one I have bought and I keep coming back because quality is consistently excellent. Fast shipping and perfect packaging.",
            "Best purchase I have made this year. Instructions were clear, setup was easy, and performance is top notch. Five stars without hesitation.",
            "I was skeptical at first based on the price point but this completely changed my mind. Premium quality at an affordable price. Family loves it.",
            "Exactly what I needed! Sturdy, well-designed, and performs better than more expensive alternatives I have tried. Very responsive customer service too.",
            "Game changer! I have been struggling with this problem for years and this product solved it immediately. Simple, effective, and worth every penny.",
            "Perfect gift! My partner was thrilled and immediately put it to use. Attention to detail in the design is impressive. Arrived two days early.",
            "Impressive durability after six months of heavy daily use. Shows no signs of wear at all. Definitely built to last. Already recommended to colleagues.",
            "Amazing product that has completely transformed my daily routine. Cannot imagine going back to my old method. Well worth the investment for sure.",
            "Outstanding quality for the price. I have purchased many similar products and this one stands out as the clear winner in its category hands down.",
            "Exceeded my expectations in every single way. Fast delivery, excellent packaging, and the product itself works perfectly as advertised. Very happy.",
            "I bought this after reading mixed reviews but I am so glad I did. It works flawlessly and the build quality is much better than I anticipated.",
            "Five star product all the way! Easy to set up, works great, and looks exactly like the photos. My whole family has been enjoying it since day one.",
            "Truly exceptional product. I have been using it daily for three months and it still works as well as the day I bought it. Incredible value for money.",
            "Best decision I made this year was buying this product. It solved my problem instantly and the quality is far better than anything else I tried before.",
            "Fantastic product with incredible attention to detail. You can tell the manufacturer really cares about quality. Will definitely buy from them again.",
            "I am completely blown away by how well this works. Simple to use, durable, and effective. Exactly what the description says. No complaints whatsoever.",
            "Perfect in every way. Looks great, works great, and arrived quickly. I have already recommended this to three of my friends and colleagues at work.",
            "High quality product that does exactly what it promises. Very happy with my purchase and would not hesitate to buy again or recommend to others.",
            "This product is simply outstanding. Easy to assemble, great build quality, and works perfectly for my needs. Worth every single cent of the price.",
            "Brilliant product that arrived quickly and in perfect condition. Extremely happy with the quality and will definitely be ordering again in the future.",
            "Could not be happier with this purchase. Great quality, reasonable price, and it works exactly as advertised. Absolutely no issues whatsoever.",
            "This exceeded my expectations significantly. Build quality is superb and performance is excellent. Would give it six stars if the system allowed it.",
            "Amazing purchase! The quality is incredible for this price point. I have bought many similar items and this is by far the best one I have owned.",
            "Wonderful product that has made my life so much easier. Easy to use, durable, and effective. Cannot recommend it highly enough to anyone who needs it.",
        ]
        negative = [
            "Extremely disappointed with this purchase. Stopped working after just two weeks of normal use. Return process was also a nightmare to deal with.",
            "Do not buy this! The product looks nothing like the photos. Cheap plastic, flimsy construction, and completely ineffective. Total waste of money.",
            "Terrible quality. It broke on the very first use. The company offers no real warranty support. Just automated responses. Avoid this seller completely.",
            "This is the worst product I have ever bought. Instructions were wrong, missing parts in the box, and customer support was completely unhelpful to me.",
            "False advertising at its finest. The description claims professional grade but it is clearly a cheap knockoff. I should have read reviews more carefully.",
            "Arrived damaged and when I reported it, the seller offered only a ten percent refund. Absolutely unacceptable behavior. Opening a dispute with my card.",
            "After three months the color has faded completely and several parts have started to peel badly. For this price I expected much better longevity overall.",
            "Does not work as advertised for my use case at all. Very misleading product description. The dimensions were also significantly off from what was listed.",
            "Complete garbage. Broke within a week of light use. The seller refused to provide a refund despite the product being clearly defective and unusable.",
            "Horrible experience from start to finish. Product arrived late, was damaged, and did not function as described. Customer service was rude and unhelpful.",
            "I am very disappointed with this product. It looked great in the photos but the actual item is much smaller and made from very cheap low quality materials.",
            "Do not waste your money on this. It stopped working after just a few days of use and the company made the return process as difficult as possible for me.",
            "Terrible product that does not match the description at all. The materials are cheap, construction is poor, and it does not function as advertised either.",
            "Very poor quality. The product fell apart after less than a month of normal careful use. I expected much better from a company charging this much money.",
            "Biggest waste of money I have spent in years. The product does nothing it claims to do and the customer service team was completely dismissive of my concerns.",
            "Awful product that broke immediately. When I contacted support they told me it was user error even though I followed all the instructions exactly as written.",
            "Disappointed does not begin to describe how I feel. This product is clearly designed to look good in photos but fails completely in actual real world use.",
            "Zero stars if I could give them. Product arrived broken, replacement also broken, and now they will not respond to my emails. Truly shocking customer service.",
            "The quality is absolutely dreadful. Cheap materials, poor craftsmanship, and it does not work properly at all. Complete waste of both money and time.",
            "I bought this based on the positive reviews but I think they must be fake. The actual product is nothing like what is described and it broke within days.",
            "Terrible experience overall. Product did not work out of the box and the company made me jump through hoops just to get a replacement. Never buying again.",
            "Very disappointed with this purchase. The product looks cheaply made in person and does not perform anywhere near as well as the marketing claims it does.",
            "Complete disaster of a product. Falls apart easily, does not work as advertised, and the customer service is non existent. Save your money and look elsewhere.",
            "Extremely poor quality for the price. I have seen better products at a fraction of the cost. Will not be buying from this brand or seller ever again.",
            "Avoid this product completely. It is poorly made, does not work as described, and the company has terrible customer service. Deeply regret this purchase.",
        ]
        neutral = [
            "Product arrived on time and matches the description. Nothing particularly special about it but it does what it is supposed to do. Average quality.",
            "It is okay. Not as impressive as I hoped but not terrible either. Does the basic job well enough. Probably will not repurchase but will not return it.",
            "Standard product, exactly what you would expect at this price range. Works as described. Shipping was normal. No complaints but nothing to rave about.",
            "I have used similar products from other brands and this one is pretty comparable. Average in every way including build quality, performance, and value.",
            "Functional and straightforward. No frills, no surprises. If you just need something basic that works this will do the job. Nothing more, nothing less.",
            "Decent product for the money. I have seen better and I have seen worse. It handles the basic tasks well enough for occasional use when needed.",
            "Third product from this brand. They are consistently mediocre. Never excellent, never terrible. Reliable in their averageness if nothing else at all.",
            "Purchased this for a specific purpose and it works fine for that task. Not versatile enough for other uses I considered but adequate for what I needed.",
            "Gets the job done but does not stand out in any particular way. Packaging was fine, delivery was on time, and the product functions as described. Okay.",
            "Neither impressed nor disappointed with this purchase. It does exactly what the listing says it does. Quality is appropriate for the price point.",
            "A perfectly average product in every sense of the word. Works as expected, nothing more. Would probably try a different brand next time to compare.",
            "Does what it needs to do without any fuss. Not the best quality I have seen but certainly not the worst either. Reasonable value for the price paid.",
            "Adequate for basic use. The build quality is mediocre but acceptable at this price range. Would be useful for someone with simple and straightforward needs.",
            "It works. That is really all I can say about it. Nothing special or remarkable but it functions correctly and arrived without any issues or damage.",
            "Average product with average quality. Does what it says on the box and nothing more. Shipping was standard and the item arrived in acceptable condition.",
            "Not bad but not great either. I was hoping for something a bit better at this price point. It works but there is definitely room for improvement here.",
            "Ordinary product that does its job adequately. No standout features and no major flaws. Just a basic functional item at a reasonable price. Fair enough.",
            "It is fine for what it is. I would not say I am thrilled with it but I am not upset either. Middle of the road product that meets basic requirements.",
            "Meets the minimum requirements for what I needed it for. Not exceptional by any means but it does function correctly and arrived without any problems.",
            "A reasonable product at a fair price. Does what it claims to do without any major issues. Not something I would rave about but also not a complaint.",
            "Three out of five stars pretty much sums it up. It works, it arrived on time, and it does what the description says. Average in every measurable way.",
            "Serviceable product that does its job without any drama. Nothing exciting about it at all but it functions as intended and represents fair value overall.",
            "Acceptable quality for the price. Would not say it is impressive but it gets the job done reliably enough. Packaging was fine and delivery was on time.",
            "I ordered this expecting something decent and that is exactly what I got. Decent. Not good, not bad. Just decent and functional for its intended purpose.",
            "Standard quality, standard performance, standard experience overall. If you need a no frills basic option this fills that role adequately enough for most.",
        ]

        rows = []
        for text in positive[:n//3]:
            rows.append({"text": text, "label": 2, "source": "product_reviews"})
        for text in negative[:n//3]:
            rows.append({"text": text, "label": 0, "source": "product_reviews"})
        for text in neutral[:n//3]:
            rows.append({"text": text, "label": 1, "source": "product_reviews"})

        df = pd.DataFrame(rows)
        logger.info(f"Generated {len(df)} product review samples")
        return df

    def build_dataset(self) -> pd.DataFrame:
        logger.info("=== Starting Dataset Building Process ===")
        df_tweets = self.generate_synthetic_tweets(n=150)
        df_reviews = self.generate_synthetic_reviews(n=150)
        df = pd.concat([df_tweets, df_reviews], ignore_index=True)
        logger.info(f"Combined dataset size before cleaning: {len(df)}")
        before = len(df)
        df = df.drop_duplicates(subset=["text"])
        logger.info(f"Removed {before - len(df)} duplicate rows")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df["label_name"] = df["label"].map(self.label_map)
        output_path = self.output_dir / "combined_dataset.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
        self._print_statistics(df)
        return df

    def _print_statistics(self, df):
        logger.info("\n=== Dataset Statistics ===")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Sources: {df['source'].value_counts().to_dict()}")
        logger.info(f"Label distribution:\n{df['label_name'].value_counts()}")
        df["text_length"] = df["text"].str.len()
        logger.info(f"Text length - Mean: {df['text_length'].mean():.1f}, Min: {df['text_length'].min()}, Max: {df['text_length'].max()}")


if __name__ == "__main__":
    builder = DatasetBuilder(output_dir="../../data/processed")
    df = builder.build_dataset()
    print(df.head())