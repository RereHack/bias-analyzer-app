# --- Import ---
import re
from collections import Counter
import nltk 
from transformers import pipeline
import spacy

# --- spaCy Download ---
try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    raise

sentiment_pipeline_en = None
sentiment_pipeline_ar = None

def get_english_pipeline():
    global sentiment_pipeline_en
    if sentiment_pipeline_en is None:
        print("Loading English sentiment analysis model...")
        sentiment_pipeline_en = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        print("English model loaded.")
    return sentiment_pipeline_en

def get_arabic_pipeline():
    global sentiment_pipeline_ar
    if sentiment_pipeline_ar is None:
        print("Loading Arabic sentiment analysis model (CAMeL-Lab)...")
        sentiment_pipeline_ar = pipeline("sentiment-analysis", model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")
        print("Arabic model loaded.")
    return sentiment_pipeline_ar

# --- NLTK Download ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK 'punkt' and 'punkt_tab' data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

# --- Normalization Arabic Words ---
def normalize_arabic(word):
    if word.startswith(('و', 'ف', 'ب', 'ك', 'ل')) and len(word) > 2:
        word = word[1:]
    
    if word.startswith('ال') and len(word) > 2:
        word = word[2:]
        
    return word

# --- (Lexicons) ---
# 1. LEXICONS: (ENGLISH)
MALE_WORDS_EN = {
    'he', 'him', 'his', 'himself', 'man', 'men', 'boy', 'boys', 'gentleman', 'gentlemen',
    'father', 'fathers', 'son', 'sons', 'husband', 'husbands', 'boyfriend',
    'king', 'prince', 'actor', 'actors', 'chairman', 'brother', 'brothers'
}
FEMALE_WORDS_EN = {
    'she', 'her', 'hers', 'herself', 'woman', 'women', 'girl', 'girls', 'lady', 'ladies',
    'mother', 'mothers', 'daughter', 'daughters', 'wife', 'wives', 'girlfriend',
    'queen', 'princess', 'actress', 'actresses', 'chairwoman', 'sister', 'sisters'
}
RELIGIOUS_TERMS_EN = {
    'islam', 'islamic', 'muslim', 'muslims', 'quran', 'allah', 'muhammad',
    'christianity', 'christian', 'christians', 'bible', 'church', 'jesus', 'christ',
    'judaism', 'jewish', 'jews', 'torah', 'synagogue', 'rabbi',
    'buddhism', 'buddhist', 'buddhists',
    'hinduism', 'hindu', 'hindus'
}
NATIONALITY_ETHNICITY_TERMS_EN = {
    'asian', 'asians', 'african', 'africans', 'european', 'europeans', 'american', 'americans', 
    'arab', 'arabs', 'arabic', 
    'saudi', 'saudis', 'egyptian', 'egyptians', 'chinese', 'indian', 'indians', 'russian', 'russians', 
    'mexican', 'mexicans', 'german', 'germans', 'japanese',
    'white', 'black', 'hispanic', 'latino', 'latina',
    'middle', 'eastern'
}
MALE_CODED_JOBS_EN = {
    'engineer', 'engineers', 'doctor', 'doctors', 'physician', 'manager', 'managers', 
    'pilot', 'pilots', 'programmer', 'programmers', 'scientist', 'scientists',
    'ceo', 'boss', 'chairman', 'president', 'officer', 'developer', 'developers', 'architect', 'architects'
}
FEMALE_CODED_JOBS_EN = {
    'nurse', 'nurses', 'secretary', 'secretaries', 'teacher', 'teachers', 
    'receptionist', 'receptionists', 'housewife', 'housewives', 'assistant', 'assistants',
    'maid', 'nanny', 'chairwoman', 'librarian', 'librarians', 'designer', 'designers'
}

# 2. LEXICONS (ARABIC)
MALE_WORDS_AR = {
    'هو', 'رجل', 'رجال', 'ولد', 'أولاد', 'أب', 'آباء', 'أبناء', 'ابن', 'زوج', 'أزواج', 
    'ملك', 'أمير', 'ممثل', 'رئيس', 'أخ', 'إخوة', 'فتى', 'شاب', 'شباب'
}
FEMALE_WORDS_AR = {
    'هي', 'امرأة', 'نساء', 'بنت', 'بنات', 'أم', 'أمهات', 'ابنة', 'بنت',
    'زوجة', 'زوجات', 'ملكة', 'أميرة', 'ممثلة', 'رئيسة', 'أخت', 'أخوات', 'فتاة', 'شابة'
}
RELIGIOUS_TERMS_AR = {
    'إسلام', 'مسلم', 'مسلمة', 'مسلمين', 'مسلمون', 'مسلمات', 'قرآن', 'الله', 'محمد',
    'مسيحية', 'مسيحي', 'مسيحية', 'مسيحيين', 'مسيحيون', 'مسيحيات', 'إنجيل', 'كنيسة', 'يسوع', 'مسيح',
    'يهودية', 'يهودي', 'يهودية', 'يهود', 'توراة', 'كنيس',
    'بوذية', 'بوذي', 'هندوسية', 'هندوسي'
}
NATIONALITY_ETHNICITY_TERMS_AR = {
    'آسيوي', 'آسيوية', 'أفريقي', 'أفريقية', 'أوروبي', 'أوروبية', 'أمريكي', 'أمريكية', 'أمريكيون', 'أمريكيين', 'أمريكيات',
    'عربي', 'عربية', 'عرب', 
    'سعودي', 'سعودية', 'سعوديون', 'سعوديين', 'سعوديات', 
    'مصري', 'مصرية', 'مصريون', 'مصريين', 'مصريات', 
    'صيني', 'صينية', 'هندي', 'هندية', 'هنود', 'روسي', 'روسية', 'مكسيكي', 'مكسيكية', 'ألماني', 'ألمانية', 'ياباني', 'يابانية', 
    'أبيض', 'أسود', 'لاتيني'
}
MALE_CODED_JOBS_AR = {
    'مهندس', 'مهندسون', 'مهندسين', 'طبيب', 'أطباء', 'مدير', 'مدراء', 'مديرين', 'طيار', 'طيارين', 
    'مبرمج', 'مبرمجون', 'مبرمجين', 'عالم', 'علماء', 'رئيس', 'رؤساء', 'دكتور', 'دكاترة',
    'مطور', 'مطورون', 'مطورين', 'ضابط', 'ضباط', 'باحث', 'باحثون', 'مصمم', 'مصممون', 'ميكانيكي', 'سائق', 'سائقون', 'جراح'
}
FEMALE_CODED_JOBS_AR = {
    'ممرضة', 'ممرضات', 'سكرتيرة', 'سكرتيرات', 'معلمة', 'معلمات', 'مدرسة', 'مدرسات',
    'موظفة', 'موظفات', 'استقبال', 'ربة', 'منزل', 'مساعدة', 'مساعدات', 'خادمة', 'خادمات', 'دكتورة', 'دكتورات',
    'طبيبة', 'طبيبات', 'مهندسة', 'مهندسات', 'مديرة', 'مديرات', 
    'باحثة', 'باحثات', 'مصممة', 'مصممات', 'مكتبية', 'خياطة'
}

# --- Bias Score Function ---
def calculate_bias_score(count_a, count_b):
    total = count_a + count_b
    if total == 0:
        return 0
    score = (abs(count_a - count_b) / total) * 100
    return int(score)


def analyze_text(text, language='English'):
    # --- ARABIC Input ---
    if language == 'Arabic':
        pipeline_ar = get_arabic_pipeline()#Analysis Input
        sentences = nltk.sent_tokenize(text)#Spliting Input by NLTK
        
        male_words_found = []
        female_words_found = []
        religion_words_found = []
        nat_eth_words_found = []
        male_jobs_found = []
        female_jobs_found = []
        sentiments = {
            'gender': {'positive': 0, 'negative': 0, 'neutral': 0},
            'occupation': {'positive': 0, 'negative': 0, 'neutral': 0},
            'religion': {'positive': 0, 'negative': 0, 'neutral': 0},
            'nationality': {'positive': 0, 'negative': 0, 'neutral': 0}
        }
        for sentence in sentences:
            try:
                sentiment_result = pipeline_ar(sentence)
                sentiment_label = sentiment_result[0]['label'] 
            except Exception:
                sentiment_label = "neutral"
            found_in_sentence = {'gender': False, 'occupation': False, 'religion': False, 'nationality': False}
            words_in_sentence = nltk.word_tokenize(sentence)
            for word in words_in_sentence:
                normalized_word = normalize_arabic(word) 
                if normalized_word in MALE_WORDS_AR:
                    male_words_found.append(normalized_word)
                    found_in_sentence['gender'] = True
                elif normalized_word in FEMALE_WORDS_AR:
                    female_words_found.append(normalized_word)
                    found_in_sentence['gender'] = True
                if normalized_word in MALE_CODED_JOBS_AR:
                    male_jobs_found.append(normalized_word)
                    found_in_sentence['occupation'] = True
                elif normalized_word in FEMALE_CODED_JOBS_AR:
                    female_jobs_found.append(normalized_word)
                    found_in_sentence['occupation'] = True
                if normalized_word in RELIGIOUS_TERMS_AR:
                    religion_words_found.append(normalized_word)
                    found_in_sentence['religion'] = True
                if normalized_word in NATIONALITY_ETHNICITY_TERMS_AR:
                    nat_eth_words_found.append(normalized_word)
                    found_in_sentence['nationality'] = True
            if found_in_sentence['gender']:
                sentiments['gender'][sentiment_label] += 1
            if found_in_sentence['occupation']:
                sentiments['occupation'][sentiment_label] += 1
            if found_in_sentence['religion']:
                sentiments['religion'][sentiment_label] += 1
            if found_in_sentence['nationality']:
                sentiments['nationality'][sentiment_label] += 1
        
        male_counts = Counter(male_words_found)
        female_counts = Counter(female_words_found)
        religion_counts = Counter(religion_words_found)
        nat_eth_counts = Counter(nat_eth_words_found)
        male_job_counts = Counter(male_jobs_found)
        female_job_counts = Counter(female_jobs_found)
        male_total = sum(male_counts.values())
        female_total = sum(female_counts.values())
        male_job_total = sum(male_job_counts.values())
        female_job_total = sum(female_job_counts.values())
        combined_male_coded = male_total + male_job_total
        combined_female_coded = female_total + female_job_total

        results = {
            'gender_analysis': {'male_count': male_total, 'female_count': female_total, 'male_words_freq': dict(male_counts), 'female_words_freq': dict(female_counts), 'sentiment_scores': sentiments['gender']},
            'religion_analysis': {'count': sum(religion_counts.values()), 'words_freq': dict(religion_counts), 'sentiment_scores': sentiments['religion']},
            'nationality_ethnicity_analysis': {'count': sum(nat_eth_counts.values()), 'words_freq': dict(nat_eth_counts), 'sentiment_scores': sentiments['nationality']},
            'occupation_analysis': {'male_job_count': male_job_total, 'female_job_count': female_job_total, 'male_job_freq': dict(male_job_counts), 'female_job_freq': dict(female_job_counts), 'sentiment_scores': sentiments['occupation']},
            'overall_scores': {'combined_bias_score': calculate_bias_score(combined_male_coded, combined_female_coded)}
        }
        return results
    # --- ENGLISH Input ---
    else:
        pipeline_en = get_english_pipeline()#Analysis Input
        doc = nlp_en(text)#Spliting Input by spaCy

        male_words_found = []
        female_words_found = []
        religion_words_found = []
        nat_eth_words_found = []
        male_jobs_found = []
        female_jobs_found = []

        sentiments = {
            'gender': {'positive': 0, 'negative': 0, 'neutral': 0},
            'occupation': {'positive': 0, 'negative': 0, 'neutral': 0},
            'religion': {'positive': 0, 'negative': 0, 'neutral': 0},
            'nationality': {'positive': 0, 'negative': 0, 'neutral': 0}
        }
        #Solve Conflict
        for sent in doc.sents:
            clauses = re.split(r', \b(but|however|whereas)\b', str(sent), flags=re.IGNORECASE)
            
            for clause in clauses:
                if not clause or clause.strip().lower() in ['but', 'however', 'whereas']:
                    continue
                
                clause_text = clause.strip()
                
                try:
                    sentiment_result = pipeline_en(clause_text)
                    sentiment_label = sentiment_result[0]['label'].lower()
                except Exception:
                    sentiment_label = "neutral"

                clause_lower = clause_text.lower()
                tokens = re.findall(r'\b\w+\b', clause_lower)
                
                found_in_sentence = {'gender': False, 'occupation': False, 'religion': False, 'nationality': False}

                for word in tokens:
                    if word in MALE_WORDS_EN:
                        male_words_found.append(word)
                        found_in_sentence['gender'] = True
                    elif word in FEMALE_WORDS_EN:
                        female_words_found.append(word)
                        found_in_sentence['gender'] = True
                    if word in MALE_CODED_JOBS_EN:
                        male_jobs_found.append(word)
                        found_in_sentence['occupation'] = True
                    elif word in FEMALE_CODED_JOBS_EN:
                        female_jobs_found.append(word)
                        found_in_sentence['occupation'] = True
                    if word in RELIGIOUS_TERMS_EN:
                        religion_words_found.append(word)
                        found_in_sentence['religion'] = True
                    if word in NATIONALITY_ETHNICITY_TERMS_EN:
                        nat_eth_words_found.append(word)
                        found_in_sentence['nationality'] = True
                
                if found_in_sentence['gender']:
                    sentiments['gender'][sentiment_label] += 1
                if found_in_sentence['occupation']:
                    sentiments['occupation'][sentiment_label] += 1
                if found_in_sentence['religion']:
                    sentiments['religion'][sentiment_label] += 1
                if found_in_sentence['nationality']:
                    sentiments['nationality'][sentiment_label] += 1

        male_counts = Counter(male_words_found)
        female_counts = Counter(female_words_found)
        religion_counts = Counter(religion_words_found)
        nat_eth_counts = Counter(nat_eth_words_found)
        male_job_counts = Counter(male_jobs_found)
        female_job_counts = Counter(female_jobs_found)
        
        male_total = sum(male_counts.values())
        female_total = sum(female_counts.values())
        male_job_total = sum(male_job_counts.values())
        female_job_total = sum(female_job_counts.values())

        combined_male_coded = male_total + male_job_total
        combined_female_coded = female_total + female_job_total

        results = {
            'gender_analysis': {'male_count': male_total, 'female_count': female_total, 'male_words_freq': dict(male_counts), 'female_words_freq': dict(female_counts), 'sentiment_scores': sentiments['gender']},
            'religion_analysis': {'count': sum(religion_counts.values()), 'words_freq': dict(religion_counts), 'sentiment_scores': sentiments['religion']},
            'nationality_ethnicity_analysis': {'count': sum(nat_eth_counts.values()), 'words_freq': dict(nat_eth_counts), 'sentiment_scores': sentiments['nationality']},
            'occupation_analysis': {'male_job_count': male_job_total, 'female_job_count': female_job_total, 'male_job_freq': dict(male_job_counts), 'female_job_freq': dict(female_job_counts), 'sentiment_scores': sentiments['occupation']},
            'overall_scores': {'combined_bias_score': calculate_bias_score(combined_male_coded, combined_female_coded)}
        }
        return results