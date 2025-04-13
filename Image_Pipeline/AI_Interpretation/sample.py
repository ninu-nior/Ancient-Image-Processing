import os
from langchain_groq import ChatGroq
import re


def remove_think_tags(text):
    """Removes everything between <think> and </think> tags, including the tags themselves."""
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()
def get_response(content,lang):
    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        groq_api_key="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",
        temperature=0.6,
    )
    
    prompt = f'''
    You are given an OCR-extracted {lang} text that is distorted. Your task is to reconstruct a meaningful passage, making reasonable assumptions where necessary., explain it in a detailed storylike way, Extract key themes, figures, and philosophical concepts, then explain their significance. Ensure the response strictly contains the reconstructed interpretation and analysis, without any additional explanations of your thought process
    {content}
    '''

    response = llm.invoke(prompt)
    print(response.content)
    cleaned=remove_think_tags(response.content)
    return cleaned# Print the response directly

# Example usage:
# get_response('''||स रिदी र२४.८६०२८०य.न ठ न7प्-ऋवितनिनिः( १ २९३ से

# | ता-२८५२ ६ स्‌ र? तपं लएण(८र स्र २३०२९८२ रररे; (|
#  ८ वाद्ये

# -२म२२३९ दप ३।त्व्यर २।€ 7२ त्ट।-रद्‌ ञ्जः खरल सुत्त
# चआद्य्तदप (मा श सा( छरी ए(तत्मानि 2 ज रे८ टार २। > र

# (चनरि४ दो रण्व दश्वी८ व्य नव्येन धे वह्देतेर रध राष्यषन्‌ |

# (पद्‌ अः-य(र-मनररिष्य ल्दैर३० २१,(३।य्‌7२८९। ख्व (२, १२,२५.

# तता त्चि च त्वतय = द = ₹२ दर श८

# ^ ०४ न हि | द 0१ शण । [वा ,-,_ हि | अ ॥ अः कन्द [| ८१८
# "४ एं 7) च

# ०८

# त

# ( ५ खरन्नि त २ ३।खत्ठव्वे यास्य त देप शी त
# गिर. र रात50८ नदर कय दर ६८ दी धनसार्कै ध (
# वसार (सिमः < व्छिमना रः! त्व 100 दिष्टः तरवा से
# व -लेवे(-ा द्धे रेल ने्तस्तेरभते-रः॥०।ए करीति( कसि ।वो

# इप्सा मं राभ टराङ्ट्‌=र$ टेरे च(न(व्वदतकप्मरसिती च

# ८ ।
# क) < /। ८

# चि! टात्‌ (२।द्दने((ना्व्रनयव्य। द्ये यन्यि योस्ज्येय। उपार ५
# | निनयनं तभरसेद्यसञदात(सेरप्तपार्नान्पाःपितानहिष्स्य पलि
# 11

# |. „^

# नप्यी ऋ तरा खद॑दत्सा मेवद (वार ३ दलन (ादिन्यिौपि ।

# इप्सा मं राभ टराङ्ट्‌=र$ टेरे च(न(व्वदतकप्मरसिती च

# ८ ।
# क) < /। ८

# चि! टात्‌ (२।द्दने((ना्व्रनयव्य। द्ये यन्यि योस्ज्येय। उपार ५
# | निनयनं तभरसेद्यसञदात(सेरप्तपार्नान्पाःपितानहिष्स्य पलि
# 11

# |. „^

# नप्यी ऋ तरा खद॑दत्सा मेवद (वार ३ दलन (ादिन्यिौपि ।

# ८ ।
# क) < /। ८

# चि! टात्‌ (२।द्दने((ना्व्रनयव्य। द्ये यन्यि योस्ज्येय। उपार ५
# | निनयनं तभरसेद्यसञदात(सेरप्तपार्नान्पाःपितानहिष्स्य पलि
# 11

# |. „^

# नप्यी ऋ तरा खद॑दत्सा मेवद (वार ३ दलन (ादिन्यिौपि ।
# चि! टात्‌ (२।द्दने((ना्व्रनयव्य। द्ये यन्यि योस्ज्येय। उपार ५
# | निनयनं तभरसेद्यसञदात(सेरप्तपार्नान्पाःपितानहिष्स्य पलि
# 11

# |. „^

# नप्यी ऋ तरा खद॑दत्सा मेवद (वार ३ दलन (ादिन्यिौपि ।

# ९
# ९


# ''')
