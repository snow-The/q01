"""雙語情感語料（繁體中文 + 英文），供 QBN × model2vec 端到端初驗使用。

設計原則
--------
1. **不依賴任何下載**：純文字檔，保證可重現、可離線執行。
2. **同時測繁體中文與英文**：這是本書的未核實項之一
   （potion-multilingual-128M 的語言標籤只有 `zh`，沒有 zh-Hant 區分）。
3. **句子簡短但有多樣性**：包含否定、轉折、程度副詞，避免模型只學到關鍵詞。
4. **類別平衡**：正負各半，讓準確率的隨機基準線是 0.50。

標籤：1 = 正面、0 = 負面。

注意：這是**教學用的迷你資料集**，不是基準測試。任何在此資料集上的結論
都必須在正式實驗中用更大的公開資料集複驗。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 繁體中文（台灣用語）
# ---------------------------------------------------------------------------
ZH_POSITIVE: list[str] = [
    "這部電影真的很好看，我從頭到尾都沒有分心。",
    "服務態度非常好，下次一定還會再來。",
    "餐點美味又便宜，cp 值很高。",
    "房間乾淨明亮，住起來很舒服。",
    "這本書寫得很精彩，我一個晚上就讀完了。",
    "老師講解得很清楚，我終於懂了。",
    "產品品質超出我的預期，非常滿意。",
    "這款手機的拍照功能很強，隨手拍都好看。",
    "客服回應迅速，問題很快就解決了。",
    "環境很安靜，適合專心工作。",
    "這間店的咖啡香醇順口，我很喜歡。",
    "程式跑得很順，完全沒有卡頓。",
    "同學們都很友善，我在這裡交到很多朋友。",
    "風景非常漂亮，值得專程來一趟。",
    "價格合理，品質也不錯，推薦給大家。",
    "這堂課的內容很實用，收穫良多。",
    "介面設計簡潔，用起來很順手。",
    "活動辦得很成功，大家都玩得很開心。",
    "這家醫院的護理師很細心，讓人放心。",
    "音樂會的演出精彩絕倫，掌聲不斷。",
    "包裝很用心，收到時完全沒有損傷。",
    "味道清爽不油膩，很適合夏天。",
    "說明書寫得很詳細，新手也能輕鬆上手。",
    "這款遊戲的劇情引人入勝，我玩了好幾輪。",
    "司機很準時，車子也很乾淨。",
    "報告的資料很紮實，委員們都給予肯定。",
    "這間民宿的主人很熱情，像回家一樣。",
    "畫質清晰，音效也很棒，值得進電影院看。",
    "這次改版讓操作變得更直覺了。",
    "維修速度很快，而且收費透明。",
    "整體來說非常值得，我會推薦給朋友。",
    "這個功能解決了我長期的困擾，太棒了。",
    "食物新鮮，份量也很足夠。",
    "網站載入速度很快，瀏覽體驗很好。",
    "教練很有耐心，講解動作很到位。",
    "這款保養品用了一週，皮膚明顯改善。",
    "流程簡單明瞭，不用跑很多趟。",
    "社團氣氛融洽，學長姐都很照顧新人。",
    "翻譯品質很好，讀起來很流暢。",
    "座位寬敞舒適，長途飛行也不覺得累。",
]

ZH_NEGATIVE: list[str] = [
    "這部電影很難看，我中途就想離場了。",
    "服務態度很差，等了半小時還沒人理。",
    "餐點又貴又難吃，完全不值得。",
    "房間有霉味，隔音也很差。",
    "這本書內容空洞，翻了幾頁就看不下去。",
    "老師講得含糊不清，我還是完全不懂。",
    "產品品質不如預期，用兩天就壞了。",
    "這款手機耗電很快，拍照也模糊。",
    "客服一直跳針，問題根本沒解決。",
    "環境非常吵雜，根本無法專心。",
    "這間店的咖啡淡而無味，不會再來。",
    "程式三不五時就當掉，非常困擾。",
    "同學之間氣氛冷淡，我沒有交到什麼朋友。",
    "風景普通，大老遠跑來有點失望。",
    "價格偏高，品質卻不成比例。",
    "這堂課的內容很空泛，浪費時間。",
    "介面複雜難懂，找個功能要找半天。",
    "活動安排混亂，大家都在空等。",
    "這家醫院的等待時間太長，讓人焦慮。",
    "音樂會的音響效果很差，聽不清楚。",
    "包裝簡陋，收到時盒子已經壓壞了。",
    "味道油膩又重鹹，吃完很不舒服。",
    "說明書寫得不清不楚，我裝了兩小時還裝不好。",
    "這款遊戲的bug很多，玩起來很煩。",
    "司機遲到很久，車上還有菸味。",
    "報告的資料錯誤百出，被委員嚴厲指正。",
    "這間民宿的主人很冷淡，感覺不受歡迎。",
    "畫質模糊，音效也很糟，不推薦。",
    "這次改版把好用的功能都拿掉了。",
    "維修拖了很久，收費還不透明。",
    "整體來說很不值得，我不會再買。",
    "這個功能反而製造更多麻煩，很難用。",
    "食物不新鮮，份量也少得可憐。",
    "網站一直轉圈圈，瀏覽體驗很差。",
    "教練態度不耐煩，動作講得不清不楚。",
    "這款保養品用了會過敏，只好停用。",
    "流程繁瑣，跑了三趟才辦好。",
    "社團氣氛很差，幹部愛理不理。",
    "翻譯錯誤很多，讀起來很痛苦。",
    "座位狹窄，長途飛行腰酸背痛。",
]

# ---------------------------------------------------------------------------
# English
# ---------------------------------------------------------------------------
EN_POSITIVE: list[str] = [
    "This movie was genuinely great and held my attention the whole time.",
    "The staff were extremely friendly and I would definitely come back.",
    "The food was delicious and affordable, excellent value for money.",
    "The room was clean and bright, very comfortable to stay in.",
    "This book is beautifully written and I finished it in one evening.",
    "The instructor explained everything clearly and I finally understood.",
    "The product quality exceeded my expectations, very satisfied.",
    "The camera on this phone is excellent, every shot looks good.",
    "Support responded quickly and resolved my issue right away.",
    "The space is quiet and perfect for focused work.",
    "The coffee here is rich and smooth, I really like it.",
    "The program runs smoothly with no lag at all.",
    "My classmates are friendly and I have made many friends here.",
    "The scenery is stunning and well worth the trip.",
    "Reasonable price and good quality, I recommend it.",
    "The course content was practical and I learned a lot.",
    "The interface is clean and easy to use.",
    "The event was well organized and everyone had a great time.",
    "The nurses at this hospital were attentive and reassuring.",
    "The concert performance was outstanding with endless applause.",
    "The packaging was careful and everything arrived undamaged.",
    "The flavor is light and not greasy, perfect for summer.",
    "The manual is detailed and even beginners can follow it.",
    "The storyline of this game is captivating, I played it several times.",
    "The driver was punctual and the car was clean.",
    "The report was solid and the committee praised it.",
    "The host of this guesthouse was warm, it felt like home.",
    "The picture quality is sharp and the sound is great, worth the cinema ticket.",
    "This redesign made everything much more intuitive.",
    "The repair was fast and the pricing was transparent.",
    "Overall it was absolutely worth it and I will recommend it to friends.",
    "This feature solved a long-standing problem for me, fantastic.",
    "The food was fresh and the portions were generous.",
    "The website loads quickly and the browsing experience is great.",
    "The coach was patient and explained every movement precisely.",
    "After a week this skincare product clearly improved my skin.",
    "The process was simple and I did not have to make multiple trips.",
    "The club atmosphere is welcoming and seniors take good care of newcomers.",
    "The translation quality is excellent and reads very smoothly.",
    "The seats were spacious and comfortable even on a long flight.",
]

EN_NEGATIVE: list[str] = [
    "This movie was terrible and I wanted to leave halfway through.",
    "The service was awful and nobody helped me for half an hour.",
    "The food was expensive and tasted bad, not worth it at all.",
    "The room smelled musty and the soundproofing was terrible.",
    "This book is empty and I gave up after a few pages.",
    "The instructor was vague and I still do not understand.",
    "The product quality fell short and it broke within two days.",
    "This phone drains battery quickly and the photos are blurry.",
    "Support kept repeating themselves and never solved my issue.",
    "The space is extremely noisy and impossible to focus in.",
    "The coffee here is weak and tasteless, I will not return.",
    "The program crashes constantly, extremely frustrating.",
    "The atmosphere among classmates is cold and I made no friends.",
    "The scenery was ordinary and the long trip was disappointing.",
    "The price is high and the quality does not match.",
    "The course content was vague and a waste of time.",
    "The interface is confusing and finding a feature takes forever.",
    "The event was chaotic and everyone was left waiting.",
    "The waiting time at this hospital was far too long and stressful.",
    "The sound system at the concert was poor and hard to hear.",
    "The packaging was flimsy and the box arrived crushed.",
    "The flavor is greasy and too salty, I felt unwell afterwards.",
    "The manual is unclear and I could not finish installing in two hours.",
    "This game has many bugs and is annoying to play.",
    "The driver arrived very late and the car smelled of smoke.",
    "The report was full of errors and the committee criticized it sharply.",
    "The host of this guesthouse was cold and I did not feel welcome.",
    "The picture quality is blurry and the sound is bad, not recommended.",
    "This redesign removed all the useful features.",
    "The repair dragged on and the pricing was not transparent.",
    "Overall it was not worth it and I will not buy again.",
    "This feature creates more trouble than it solves and is hard to use.",
    "The food was not fresh and the portions were tiny.",
    "The website keeps spinning and the browsing experience is poor.",
    "The coach was impatient and explained the movements poorly.",
    "This skincare product caused an allergic reaction so I stopped using it.",
    "The process is tedious and took three visits to complete.",
    "The club atmosphere is poor and the officers are unhelpful.",
    "There are many translation errors and it is painful to read.",
    "The seats were cramped and my back hurt on the long flight.",
]


def load_dataset() -> tuple[list[str], list[int], list[str]]:
    """回傳 (句子列表, 標籤列表, 語言標籤列表)。

    標籤：1 = 正面、0 = 負面。
    語言標籤：'zh' 或 'en'。
    """
    texts: list[str] = []
    labels: list[int] = []
    langs: list[str] = []

    for t in ZH_POSITIVE:
        texts.append(t); labels.append(1); langs.append("zh")
    for t in ZH_NEGATIVE:
        texts.append(t); labels.append(0); langs.append("zh")
    for t in EN_POSITIVE:
        texts.append(t); labels.append(1); langs.append("en")
    for t in EN_NEGATIVE:
        texts.append(t); labels.append(0); langs.append("en")

    return texts, labels, langs


if __name__ == "__main__":
    texts, labels, langs = load_dataset()
    print(f"總句數 = {len(texts)}")
    print(f"  正面 = {sum(labels)}，負面 = {len(labels) - sum(labels)}")
    print(f"  繁體中文 = {langs.count('zh')}，英文 = {langs.count('en')}")
    print(f"\n前 3 句（zh, 正面）:")
    for t in ZH_POSITIVE[:3]:
        print(f"  {t}")
    print(f"\n前 3 句（en, 負面）:")
    for t in EN_NEGATIVE[:3]:
        print(f"  {t}")
