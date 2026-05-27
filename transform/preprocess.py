import re
import hashlib
import unicodedata
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from nltk.corpus import stopwords as nltk_stopwords
from config import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

_NLTK_STOPWORDS = {}
_NLTK_LANG_MAP = {
    "arabic":     "ar", "danish":   "da", "dutch":     "nl",
    "english":    "en", "finnish":  "fi", "french":    "fr",
    "german":     "de", "greek":    "el", "hungarian": "hu",
    "indonesian": "id", "italian":  "it", "norwegian": "no",
    "portuguese": "pt", "romanian": "ro", "russian":   "ru",
    "spanish":    "es", "swedish":  "sv", "turkish":   "tr",
}

def _load_stopwords():
    global _NLTK_STOPWORDS
    if _NLTK_STOPWORDS:
        return
    for lang_name, iso in _NLTK_LANG_MAP.items():
        try:
            words = set(nltk_stopwords.words(lang_name))
            _NLTK_STOPWORDS[iso] = words
        except Exception:
            pass

_NON_LATIN_RANGES = [
    (0x0400, 0x04FF),
    (0x0600, 0x06FF),
    (0x0900, 0x097F),
    (0x4E00, 0x9FFF),
    (0x3040, 0x30FF),
    (0xAC00, 0xD7AF),
    (0x0E00, 0x0E7F),
    (0x0370, 0x03FF),
    (0x0500, 0x052F),
]

def _has_non_latin_script(text: str) -> bool:
    count = 0
    total = 0
    for ch in text[:300]:
        if ch.isalpha():
            total += 1
            cp = ord(ch)
            for lo, hi in _NON_LATIN_RANGES:
                if lo <= cp <= hi:
                    count += 1
                    break
    if total == 0:
        return False
    return (count / total) > 0.3

def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 20:
        return "unknown"

    _load_stopwords()

    if _has_non_latin_script(text):
        sample = text[:200]
        for ch in sample:
            cp = ord(ch)
            if 0x0600 <= cp <= 0x06FF:
                return "ar"
            if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF:
                return "zh"
            if 0xAC00 <= cp <= 0xD7AF:
                return "ko"
            if 0x0400 <= cp <= 0x04FF:
                return "ru"
            if 0x0900 <= cp <= 0x097F:
                return "hi"
            if 0x0E00 <= cp <= 0x0E7F:
                return "th"
            if 0x0370 <= cp <= 0x03FF:
                return "el"
        return "non-latin"

    words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()[:1000]))
    if not words:
        return "unknown"

    best_lang = "unknown"
    best_score = 0.0
    for iso, sw_set in _NLTK_STOPWORDS.items():
        hits = len(words & sw_set)
        score = hits / max(len(words), 1)
        if score > best_score:
            best_score = score
            best_lang = iso

    if best_score < 0.05:
        return "unknown"
    return best_lang


_US_STATE_ABBR = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
}

_COUNTRY_ALIASES: dict[str, str] = {
    "us": "United States",
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "united states of america": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "britain": "United Kingdom",
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "dubai": "United Arab Emirates",
    "abu dhabi": "United Arab Emirates",
    "south korea": "South Korea",
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "drc": "DR Congo",
    "congo": "DR Congo",
    "czech republic": "Czech Republic",
    "czechia": "Czech Republic",
    "new zealand": "New Zealand",
    "aotearoa": "New Zealand",
    "south africa": "South Africa",
    "saudi arabia": "Saudi Arabia",
    "ksa": "Saudi Arabia",
    "hong kong": "Hong Kong",
    "hk": "Hong Kong",
    "costa rica": "Costa Rica",
    "sri lanka": "Sri Lanka",
    "el salvador": "El Salvador",
    "puerto rico": "Puerto Rico",
    "dominican republic": "Dominican Republic",
    "trinidad": "Trinidad and Tobago",
    "trinidad and tobago": "Trinidad and Tobago",
    "deutschland": "Germany",
    "allemagne": "Germany",
    "france": "France",
    "frankreich": "France",
    "frankrijk": "France",
    "españa": "Spain",
    "espagne": "Spain",
    "spain": "Spain",
    "italia": "Italy",
    "italie": "Italy",
    "italy": "Italy",
    "nederland": "Netherlands",
    "netherlands": "Netherlands",
    "pays-bas": "Netherlands",
    "sverige": "Sweden",
    "suède": "Sweden",
    "sweden": "Sweden",
    "norge": "Norway",
    "norvège": "Norway",
    "norway": "Norway",
    "danmark": "Denmark",
    "danemark": "Denmark",
    "denmark": "Denmark",
    "suomi": "Finland",
    "finlande": "Finland",
    "finland": "Finland",
    "schweiz": "Switzerland",
    "suisse": "Switzerland",
    "switzerland": "Switzerland",
    "österreich": "Austria",
    "autriche": "Austria",
    "austria": "Austria",
    "belgique": "Belgium",
    "belgië": "Belgium",
    "belgium": "Belgium",
    "polska": "Poland",
    "pologne": "Poland",
    "poland": "Poland",
    "românia": "Romania",
    "roumanie": "Romania",
    "romania": "Romania",
    "україна": "Ukraine",
    "ukraine": "Ukraine",
    "magyarország": "Hungary",
    "hongrie": "Hungary",
    "hungary": "Hungary",
    "ελλάδα": "Greece",
    "grèce": "Greece",
    "greece": "Greece",
    "türkiye": "Turkey",
    "turkey": "Turkey",
    "turquie": "Turkey",
    "brasil": "Brazil",
    "brésil": "Brazil",
    "brazil": "Brazil",
    "argentina": "Argentina",
    "argentine": "Argentina",
    "colombia": "Colombia",
    "colombie": "Colombia",
    "chile": "Chile",
    "chili": "Chile",
    "peru": "Peru",
    "pérou": "Peru",
    "équateur": "Ecuador",
    "ecuador": "Ecuador",
    "afrique du sud": "South Africa",
    "südafrika": "South Africa",
    "nigéria": "Nigeria",
    "nigeria": "Nigeria",
    "égypte": "Egypt",
    "ägypten": "Egypt",
    "egypt": "Egypt",
    "éthiopie": "Ethiopia",
    "ethiopia": "Ethiopia",
    "arabie saoudite": "Saudi Arabia",
    "émirats arabes unis": "United Arab Emirates",
    "israël": "Israel",
    "israel": "Israel",
    "jordanie": "Jordan",
    "jordan": "Jordan",
    "singapour": "Singapore",
    "singapore": "Singapore",
    "indonésie": "Indonesia",
    "indonesia": "Indonesia",
    "malaisie": "Malaysia",
    "malaysia": "Malaysia",
    "viêt nam": "Vietnam",
    "vietnam": "Vietnam",
    "thaïlande": "Thailand",
    "thailand": "Thailand",
    "birmanie": "Myanmar",
    "myanmar": "Myanmar",
    "cambodge": "Cambodia",
    "cambodia": "Cambodia",
    "inde": "India",
    "indien": "India",
    "india": "India",
    "australie": "Australia",
    "australien": "Australia",
    "australia": "Australia",
    "nouvelle-zélande": "New Zealand",
    "chine": "China",
    "china vr": "China",
    "china": "China",
    "japon": "Japan",
    "japan": "Japan",
    "corée du sud": "South Korea",
    "südkorea": "South Korea",
    "taiwan": "Taiwan",
    "pakistan": "Pakistan",
    "bangladesh": "Bangladesh",
    "canada": "Canada",
    "mexico": "Mexico",
    "méxico": "Mexico",
    "mexique": "Mexico",
    "portugal": "Portugal",
    "ireland": "Ireland",
    "irlande": "Ireland",
    "irland": "Ireland",
    "kenya": "Kenya",
    "ghana": "Ghana",
    "philippines": "Philippines",
    "remote": "Remote",
    "worldwide": "Remote",
    "global": "Remote",
    "anywhere": "Remote",
    "partout": "Remote",
    "weltweit": "Remote",
}

_METRO_AREA_COUNTRY: list[tuple[str, str]] = [
    ("metropolitan area", "United States"),
    ("metro area", "United States"),
    ("greater", "United States"),
    (" area", "United States"),
]

_US_CITIES: set[str] = {
    "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia",
    "san antonio", "san diego", "dallas", "san jose", "austin", "jacksonville",
    "fort worth", "columbus", "charlotte", "san francisco", "indianapolis",
    "seattle", "denver", "washington", "nashville", "oklahoma city", "el paso",
    "boston", "portland", "las vegas", "memphis", "louisville", "baltimore",
    "milwaukee", "albuquerque", "tucson", "fresno", "sacramento", "mesa",
    "atlanta", "omaha", "colorado springs", "raleigh", "miami", "virginia beach",
    "minneapolis", "tampa", "new orleans", "cleveland", "bakersfield", "aurora",
    "anaheim", "honolulu", "corpus christi", "riverside", "lexington",
    "stockton", "st. louis", "pittsburgh", "anchorage", "greensboro",
    "lincoln", "plano", "orlando", "irvine", "newark", "durham",
    "st. paul", "laredo", "norfolk", "madison", "chandler", "lubbock",
    "scottsdale", "reno", "buffalo", "winston-salem", "gilbert",
    "glendale", "north las vegas", "garland", "hialeah", "baton rouge",
    "chesapeake", "irving", "fremont", "san bernardino", "boise",
    "birmingham", "rochester", "richmond", "spokane", "des moines",
    "montgomery", "modesto", "fayetteville", "tacoma", "shreveport",
    "akron", "aurora", "yonkers", "glendale", "huntington beach",
    "salt lake city", "amarillo", "huntsville", "grand rapids", "knoxville",
    "worcester", "newport news", "moreno valley", "tempe", "fontana",
    "garden grove", "brownsville", "oceanside", "providence", "santa clarita",
    "fort lauderdale", "chattanooga", "elk grove", "clarksville", "cape coral",
    "kansas city", "columbia", "hartford", "rockford", "little rock",
    "oxnard", "tallahassee", "ontario", "sioux falls", "peoria",
    "springfield", "eugene", "rancho cucamonga", "pembroke pines", "fort collins",
}

REGION_MAP: dict[str, str] = {
    "United States": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Costa Rica": "North America",
    "Panama": "North America",
    "Guatemala": "North America",
    "Honduras": "North America",
    "Nicaragua": "North America",
    "El Salvador": "North America",
    "Belize": "North America",
    "Jamaica": "North America",
    "Cuba": "North America",
    "Haiti": "North America",
    "Dominican Republic": "North America",
    "Puerto Rico": "North America",
    "Bahamas": "North America",
    "Trinidad and Tobago": "North America",

    "Brazil": "South America",
    "Argentina": "South America",
    "Chile": "South America",
    "Colombia": "South America",
    "Peru": "South America",
    "Ecuador": "South America",
    "Uruguay": "South America",
    "Paraguay": "South America",
    "Bolivia": "South America",
    "Venezuela": "South America",
    "Guyana": "South America",
    "Suriname": "South America",

    "United Kingdom": "Europe",
    "Ireland": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Spain": "Europe",
    "Portugal": "Europe",
    "Italy": "Europe",
    "Netherlands": "Europe",
    "Belgium": "Europe",
    "Luxembourg": "Europe",
    "Switzerland": "Europe",
    "Austria": "Europe",
    "Poland": "Europe",
    "Czech Republic": "Europe",
    "Slovakia": "Europe",
    "Hungary": "Europe",
    "Romania": "Europe",
    "Bulgaria": "Europe",
    "Greece": "Europe",
    "Croatia": "Europe",
    "Serbia": "Europe",
    "Slovenia": "Europe",
    "Bosnia and Herzegovina": "Europe",
    "Montenegro": "Europe",
    "North Macedonia": "Europe",
    "Albania": "Europe",
    "Ukraine": "Europe",
    "Belarus": "Europe",
    "Lithuania": "Europe",
    "Latvia": "Europe",
    "Estonia": "Europe",
    "Sweden": "Europe",
    "Norway": "Europe",
    "Denmark": "Europe",
    "Finland": "Europe",
    "Iceland": "Europe",
    "Moldova": "Europe",
    "Malta": "Europe",
    "Cyprus": "Europe",
    "Turkey": "Europe",

    "China": "East Asia",
    "Japan": "East Asia",
    "South Korea": "East Asia",
    "North Korea": "East Asia",
    "Taiwan": "East Asia",
    "Hong Kong": "East Asia",
    "Macau": "East Asia",
    "Mongolia": "East Asia",

    "Indonesia": "Southeast Asia",
    "Malaysia": "Southeast Asia",
    "Singapore": "Southeast Asia",
    "Thailand": "Southeast Asia",
    "Vietnam": "Southeast Asia",
    "Philippines": "Southeast Asia",
    "Myanmar": "Southeast Asia",
    "Cambodia": "Southeast Asia",
    "Laos": "Southeast Asia",
    "Brunei": "Southeast Asia",
    "Timor-Leste": "Southeast Asia",

    "India": "South Asia",
    "Pakistan": "South Asia",
    "Bangladesh": "South Asia",
    "Sri Lanka": "South Asia",
    "Nepal": "South Asia",
    "Bhutan": "South Asia",
    "Maldives": "South Asia",
    "Afghanistan": "South Asia",

    "Kazakhstan": "Central Asia",
    "Uzbekistan": "Central Asia",
    "Turkmenistan": "Central Asia",
    "Kyrgyzstan": "Central Asia",
    "Tajikistan": "Central Asia",

    "Saudi Arabia": "Middle East",
    "United Arab Emirates": "Middle East",
    "Qatar": "Middle East",
    "Kuwait": "Middle East",
    "Bahrain": "Middle East",
    "Oman": "Middle East",
    "Jordan": "Middle East",
    "Israel": "Middle East",
    "Lebanon": "Middle East",
    "Iraq": "Middle East",
    "Iran": "Middle East",
    "Syria": "Middle East",
    "Yemen": "Middle East",
    "Palestine": "Middle East",

    "South Africa": "Africa",
    "Nigeria": "Africa",
    "Kenya": "Africa",
    "Egypt": "Africa",
    "Morocco": "Africa",
    "Algeria": "Africa",
    "Tunisia": "Africa",
    "Libya": "Africa",
    "Ghana": "Africa",
    "Ethiopia": "Africa",
    "Tanzania": "Africa",
    "Uganda": "Africa",
    "Rwanda": "Africa",
    "Senegal": "Africa",
    "Ivory Coast": "Africa",
    "Cameroon": "Africa",
    "Zimbabwe": "Africa",
    "Botswana": "Africa",
    "Namibia": "Africa",
    "Mozambique": "Africa",
    "Sudan": "Africa",
    "DR Congo": "Africa",
    "Angola": "Africa",

    "Australia": "Oceania",
    "New Zealand": "Oceania",
    "Papua New Guinea": "Oceania",
    "Fiji": "Oceania",
    "Samoa": "Oceania",
    "Tonga": "Oceania",
    "Vanuatu": "Oceania",

    "Remote": "Remote/Global",
    "Worldwide": "Remote/Global",
    "Global": "Remote/Global",
}

JOB_LEVEL_KEYWORDS = {
    "Junior":  ["junior", "entry", "entry-level", "associate", "intern",
                "internship", "graduate", "jr", "early career", "trainee"],
    "Mid":     ["mid", "intermediate", "mid-level", "ii", " ii ", "level 2",
                "experienced", "mid level"],
    "Senior":  ["senior", "sr", "lead", "principal", "staff", "iii", " iii ",
                "level 3", "iv", "level 4", "expert", "seasoned"],
    "Manager": ["manager", "director", "head of", "vp ", "vice president",
                "chief", "cto", "cdo", "ciso", "coo", "president", "executive"],
}

JOB_CATEGORY_KEYWORDS = {
    "Data": [
        "data scientist", "data analyst", "data engineer", "data architect",
        "analytics engineer", "bi analyst", "business intelligence", "data steward",
        "data modeler", "data specialist", "data manager", "data consultant",
        "data quality", "data governance", "data ops", "dataops",
        "analytics", "reporting analyst", "insights analyst",
        "business analyst", "quantitative analyst", "decision scientist",
        "data visualization", "power bi", "tableau developer",
        "statistician", "research analyst", "operations analyst",
        "data entry", "data entry clerk", "data entry operator", "data entry specialist",
        "data input", "data clerk", "data processor", "data capture",
        "data administrator", "data coordinator",
    ],

    "Machine Learning": [
        "machine learning", "ml engineer", "ai engineer", "deep learning",
        "nlp engineer", "computer vision", "artificial intelligence", "llm engineer",
        "research scientist", "applied scientist", "ai researcher", "ml researcher",
        "generative ai", "large language model", "reinforcement learning",
        "recommendation", "mlops", "prompt engineer",
        "ai architect", "foundation model", "rag engineer",
        "ai consultant", "speech recognition", "ocr engineer",
        "autonomous systems", "robot learning",
    ],

    "Cloud": [
        "cloud engineer", "cloud architect", "solutions architect",
        "platform engineer", "infrastructure engineer",
        "aws engineer", "gcp engineer", "azure engineer",
        "cloud developer", "cloud consultant", "cloud administrator",
        "systems engineer", "systems administrator",
        "cloud operations", "cloud security", "multi-cloud",
        "virtualization", "vmware", "openstack",
    ],

    "DevOps": [
        "devops", "site reliability", "sre", "platform engineer",
        "release engineer", "ci/cd", "devsecops", "infrastructure as code",
        "kubernetes", "docker", "build engineer",
        "observability", "monitoring engineer", "automation engineer",
        "terraform", "ansible", "jenkins", "gitops",
        "container engineer",
    ],

    "FinTech": [
        "fintech", "quant", "quantitative", "algorithmic",
        "financial engineer", "risk analyst", "trading", "actuary",
        "credit analyst", "fraud analyst", "risk engineer",
        "investment analyst", "portfolio manager",
        "treasury analyst", "compliance analyst",
        "banking analyst", "wealth management",
    ],

    "Engineering": [
        "software engineer", "software developer", "backend", "frontend",
        "fullstack", "full stack", "full-stack", "mobile engineer",
        "ios", "android", "java developer", "python developer",
        "golang", "rust engineer", "scala", "kotlin",
        "react developer", "vue developer", "node developer",
        "game developer", "embedded engineer", "robotics",
        "web developer", "application developer", "api developer",
        "php developer", "ruby developer", "angular developer",
        "net developer", ".net", "c++ developer", "c# developer",
        "firmware engineer", "unity developer", "unreal developer",
        "desktop developer", "qt developer", "systems programmer",
        "software architect", "technical architect",
    ],

    "Management": [
        "manager", "director", "head of", "vp of",
        "chief", "engineering manager", "product manager",
        "project manager", "scrum master", "program manager",
        "delivery manager", "agile coach", "technical lead",
        "team lead", "tech lead",
        "cto", "ceo", "coo", "cio",
        "product owner", "business manager",
        "operations manager", "strategy manager",
    ],

    "Design": [
        "ux", "ui ", "designer", "product designer",
        "graphic designer", "visual designer", "interaction designer",
        "ux researcher", "design researcher",
        "motion designer", "3d designer", "illustrator",
        "creative director", "brand designer",
        "industrial designer", "game artist",
        "ui/ux", "ux writer",
    ],

    "Security": [
        "cybersecurity", "security engineer", "network engineer",
        "penetration tester", "soc analyst", "information security",
        "security analyst", "threat analyst", "vulnerability",
        "incident response", "ethical hacker",
        "red team", "blue team", "iam engineer",
        "security consultant", "forensics",
        "application security", "cloud security",
    ],

    "Database": [
        "database administrator", "dba", "sql developer",
        "etl developer", "data warehouse", "database engineer",
        "database developer", "data pipeline",
        "big data engineer", "hadoop", "spark engineer",
        "snowflake", "bigquery", "postgresql",
        "oracle dba", "mysql dba",
    ],

    "Networking": [
        "network administrator", "network architect",
        "network operations", "telecommunications",
        "wireless engineer", "voip engineer",
        "ccna", "ccnp", "network support",
        "network specialist",
    ],

    "QA & Testing": [
        "qa engineer", "test engineer", "software tester",
        "automation tester", "sdet", "quality assurance",
        "manual tester", "performance tester",
        "test automation", "uat tester",
    ],

    "Product": [
        "product owner", "associate product manager",
        "growth product manager", "technical product manager",
        "product operations", "product strategist",
        "product analyst",
    ],

    "Marketing": [
        "digital marketing", "seo", "sem", "content marketing",
        "growth hacker", "performance marketing",
        "marketing analyst", "brand manager",
        "social media manager", "copywriter",
        "email marketing", "affiliate marketing",
        "community manager",
    ],

    "Sales": [
        "sales executive", "account executive",
        "business development", "sales manager",
        "inside sales", "outside sales",
        "sales representative", "account manager",
        "customer success", "partnership manager",
        "solution consultant", "pre sales",
    ],

    "Human Resources": [
        "hr", "human resources", "recruiter",
        "talent acquisition", "people operations",
        "hr business partner", "compensation analyst",
        "learning and development", "organizational development",
        "technical recruiter", "hr generalist", "hr specialist",
        "hr coordinator", "hr admin", "hr manager", "hr analyst",
        "payroll", "talent management", "workforce planning",
        "employee relations", "onboarding specialist",
    ],

    "Customer Support": [
        "customer support", "customer service",
        "technical support", "help desk",
        "support engineer", "call center",
        "client success", "service desk",
    ],

    "Operations": [
        "operations analyst", "operations manager",
        "business operations", "supply chain",
        "logistics", "procurement",
        "inventory analyst", "warehouse manager",
        "process improvement", "warehouse", "dispatch",
        "warehouse clerk", "warehouse associate", "warehouse operator",
        "logistics coordinator", "logistics specialist", "logistics analyst",
        "fulfillment", "fulfillment associate", "shipping clerk",
        "receiving clerk", "material handler", "inventory clerk",
        "supply chain coordinator", "delivery coordinator",
        "distribution", "transport coordinator", "fleet coordinator",
    ],

    "Healthcare": [
        "doctor", "nurse", "pharmacist",
        "medical assistant", "healthcare analyst",
        "clinical researcher", "radiologist",
        "therapist", "surgeon",
        "dentist", "public health",
    ],

    "Education": [
        "teacher", "lecturer", "professor",
        "curriculum developer", "instructional designer",
        "academic advisor", "tutor",
        "education consultant",
    ],

    "Legal": [
        "lawyer", "attorney", "legal counsel",
        "paralegal", "compliance officer",
        "legal assistant", "corporate counsel",
    ],

    "Media & Communication": [
        "journalist", "editor", "producer",
        "videographer", "photographer",
        "public relations", "communications specialist",
        "news anchor", "podcast producer",
    ],

    "Creative": [
        "animator", "video editor", "sound designer",
        "music producer", "writer", "author",
        "screenwriter", "creative strategist",
    ],

    "Science & Research": [
        "scientist", "chemist", "physicist",
        "biologist", "research assistant",
        "lab technician", "research fellow",
        "environmental scientist",
    ],

    "Manufacturing": [
        "manufacturing engineer", "production engineer",
        "quality control", "plant manager",
        "mechanical engineer", "electrical engineer",
        "industrial engineer", "process engineer",
    ],

    "Construction": [
        "civil engineer", "architect",
        "construction manager", "surveyor",
        "urban planner", "site engineer",
        "structural engineer",
    ],

    "Energy": [
        "oil and gas", "renewable energy",
        "solar engineer", "energy analyst",
        "power systems engineer",
        "petroleum engineer",
    ],

    "Government": [
        "policy analyst", "public administration",
        "government relations", "diplomat",
        "civil servant", "urban policy",
    ],

    "Hospitality": [
        "hotel manager", "chef", "cook",
        "restaurant manager", "barista",
        "event coordinator", "tour guide",
        "hospitality specialist",
    ],

    "Retail": [
        "store manager", "merchandiser",
        "cashier", "retail associate",
        "buyer", "category manager",
        "ecommerce specialist",
    ],

    "Transportation": [
        "driver", "pilot", "flight attendant",
        "air traffic controller", "shipping coordinator",
        "fleet manager", "rail operator",
    ],

    "Agriculture": [
        "agricultural engineer", "farmer",
        "agronomist", "food scientist",
        "livestock manager",
    ],

    "Real Estate": [
        "real estate agent", "property manager",
        "broker", "leasing consultant",
        "facilities manager",
    ],

    "Consulting": [
        "consultant", "strategy consultant",
        "management consultant", "it consultant",
        "business consultant", "solutions consultant",
    ],

    "Blockchain & Web3": [
        "blockchain developer", "smart contract",
        "solidity developer", "web3 engineer",
        "crypto analyst", "defi",
        "nft", "ethereum developer",
    ],

    "Game Development": [
        "game designer", "level designer",
        "technical artist", "game programmer",
        "unity engineer", "unreal engineer",
        "gameplay engineer",
    ],

    "AR/VR": [
        "ar developer", "vr developer",
        "xr engineer", "mixed reality",
        "metaverse", "spatial computing",
    ],

    "IoT": [
        "iot engineer", "edge computing",
        "sensor engineer", "hardware engineer",
        "embedded systems",
    ],

    "Semiconductor": [
        "asic engineer", "fpga engineer",
        "chip designer", "verification engineer",
        "vlsi engineer", "semiconductor engineer",
    ],

    "BioTech": [
        "bioinformatics", "genomics",
        "biomedical engineer", "biotech researcher",
        "drug discovery",
    ],

    "Aviation": [
        "aerospace engineer", "aircraft mechanic",
        "avionics engineer", "flight operations",
    ],

    "Maritime": [
        "marine engineer", "naval architect",
        "ship captain", "deck officer",
        "port operations",
    ],

    "Nonprofit": [
        "ngo", "fundraising", "grant writer",
        "program coordinator", "social worker",
        "community outreach",
    ],

    "Administration": [
        "administrative assistant", "admin assistant", "office administrator",
        "executive assistant", "personal assistant", "secretary",
        "receptionist", "clerical", "office coordinator", "office manager",
        "front desk", "administrative coordinator", "administrative officer",
        "administrative staff", "administrative specialist",
        "general affairs", "office support", "document controller",
        "records manager", "filing clerk",
    ],

    "Finance": [
        "accountant", "accounting", "bookkeeper", "controller",
        "auditor", "financial analyst", "finance analyst",
        "tax analyst", "tax accountant", "tax specialist",
        "finance manager", "treasury", "budget analyst",
        "cost analyst", "financial controller", "cfo",
        "accounts payable", "accounts receivable", "finance officer",
        "financial reporting", "internal audit",
    ],
}

JOB_CATEGORY_KEYWORDS_MULTILINGUAL = {
    "Data": [
        "datos", "analista de datos", "ingeniero de datos", "entrada de datos",
        "analisis de datos", "ciencia de datos",
        "datos entrada", "captura de datos",
        "analis data", "ilmuwan data", "insinyur data", "entri data",
        "数据", "数据分析", "数据工程师", "数据科学家", "数据录入",
    ],
    "Engineering": [
        "desarrollador", "programador", "ingeniero de software",
        "desarrollo de software", "desarrollador web", "backend", "frontend",
        "pengembang", "programmer", "insinyur perangkat lunak",
        "perangkat lunak", "rekayasa perangkat lunak",
        "软件工程师", "开发工程师", "程序员", "前端", "后端", "全栈",
    ],
    "Human Resources": [
        "recursos humanos", "gestión humana", "reclutador", "talento humano",
        "capital humano", "selección de personal", "nómina",
        "sumber daya manusia", "sdm", "rekrutmen", "perekrut",
        "manajemen talenta", "penggajian",
        "人力资源", "人资", "招聘", "人事", "猎头", "hr",
    ],
    "Management": [
        "gerente", "director", "jefe de", "coordinador", "supervisor",
        "manajer", "direktur", "kepala", "koordinator", "supervisor",
        "经理", "总监", "主管", "总裁", "副总", "负责人", "项目经理",
    ],
    "Operations": [
        "logística", "logistica", "almacén", "almacen", "bodega",
        "inventario", "cadena de suministro", "despacho", "distribución",
        "operaciones", "auxiliar de bodega", "auxiliar de despacho",
        "logistik", "operasional", "gudang", "pengadaan", "distribusi",
        "运营", "物流", "仓库", "供应链", "配送", "仓储",
    ],
    "Customer Support": [
        "atención al cliente", "atencion al cliente", "servicio al cliente",
        "soporte", "atención ciudadana", "atención al ciudadano",
        "layanan pelanggan", "dukungan pelanggan", "customer service",
        "客服", "客户服务", "售后", "用户支持",
    ],
    "Sales": [
        "ventas", "ejecutivo de ventas", "representante de ventas",
        "desarrollo de negocios", "comercial", "vendedor",
        "penjualan", "sales", "tenaga penjual", "pengembangan bisnis",
        "销售", "业务", "商务", "销售代表", "销售经理",
    ],
    "Marketing": [
        "marketing", "mercadeo", "publicidad", "comunicaciones",
        "marketing digital", "redes sociales",
        "pemasaran", "marketing", "iklan", "promosi", "media sosial",
        "市场", "营销", "推广", "广告", "品牌", "市场营销",
    ],
    "Education": [
        "docente", "profesor", "maestro", "tutor", "capacitador",
        "formador", "instructor", "catedrático",
        "guru", "dosen", "pengajar", "instruktur", "pelatih",
        "教师", "讲师", "教授", "老师", "培训师", "辅导",
    ],
    "Healthcare": [
        "médico", "medico", "enfermero", "enfermera", "farmacéutico",
        "farmaceutico", "terapeuta", "paramédico",
        "dokter", "perawat", "apoteker", "terapis", "bidan",
        "医生", "护士", "药剂师", "医师", "治疗师",
    ],
    "Finance": [
        "contable", "contador", "financiero", "tesorería", "auditor",
        "contabilidad", "finanzas", "nómina contable",
        "akuntan", "akuntansi", "keuangan", "bendahara", "audit",
        "会计", "财务", "审计", "出纳", "财务分析", "税务",
    ],
    "Design": [
        "diseñador", "diseño", "diseño gráfico", "diseño ux", "diseño ui",
        "desainer", "desain", "desain grafis",
        "设计师", "设计", "平面设计", "ui设计", "ux设计",
    ],
    "Administration": [
        "administrativo", "administración", "asistente administrativo",
        "auxiliar administrativo", "secretaria", "recepcionista",
        "administrasi", "admin", "staf administrasi", "sekretaris",
        "resepsionis", "tata usaha",
        "行政", "秘书", "助理", "文员", "行政助理", "前台",
    ],
    "Manufacturing": [
        "manufactura", "producción", "calidad", "técnico", "operario",
        "mantenimiento", "mecánico",
        "manufaktur", "produksi", "kualitas", "teknisi", "operator",
        "perawatan",
        "制造", "生产", "质量", "技术员", "操作员", "维修",
    ],
    "Retail": [
        "cajero", "tienda", "comercio", "vendedor", "almacenista", "caja",
        "kasir", "toko", "perdagangan", "pramuniaga",
        "收银", "零售", "店员", "门店",
    ],
    "Security": [
        "seguridad", "vigilante", "guardia de seguridad",
        "keamanan", "satpam", "security",
        "安保", "安全", "保安",
    ],
    "Construction": [
        "construcción", "obra", "ingeniero civil", "arquitecto",
        "konstruksi", "bangunan", "insinyur sipil", "arsitek",
        "建筑", "土木", "施工", "建造",
    ],
    "Transportation": [
        "conductor", "chofer", "piloto", "transporte", "repartidor",
        "sopir", "pengemudi", "pilot", "transportasi",
        "司机", "驾驶", "快递", "配送员",
    ],
}

JOB_CATEGORY_PARTIAL_KEYWORDS = {
    "Engineering": ["developer", "programmer", "coder"],
    "Design": ["designer"],
    "Human Resources": ["recruiter", "sourcer"],
    "Marketing": ["marketer"],
    "Finance": ["accountant", "bookkeeper", "auditor"],
    "Transportation": ["driver", "chauffeur"],
    "Consulting": ["consultant"],
    "Education": ["teacher", "lecturer", "instructor", "tutor"],
    "Healthcare": ["nurse", "doctor", "physician", "therapist", "dentist"],
    "Legal": ["lawyer", "attorney", "paralegal"],
    "QA & Testing": ["tester"],
    "Data": ["analyst"],
}

SALARY_BOUNDS_BY_LEVEL = {
    "Junior": (15000, 150000),
    "Mid": (25000, 250000),
    "Senior": (40000, 500000),
    "Manager": (50000, 700000),
    "Unknown": (10000, 700000),
}

PAY_PERIOD_MULTIPLIER = {
    "hourly": 2080,
    "hour": 2080,
    "daily": 260,
    "day": 260,
    "weekly": 52,
    "week": 52,
    "monthly": 12,
    "month": 12,
    "yearly": 1,
    "year": 1,
    "annual": 1,
    "annually": 1,
}

HOURLY_SALARY_THRESHOLD = 500


def clean_text(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    if not text.strip():
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s\.,\-\+\#\/\(\)\:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def clean_text_ner(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    if not text.strip():
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<li\b[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s\.,\-\+\#\/\(\)\:\n]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_date(val) -> pd.Timestamp:
    if val is None:
        return pd.NaT
    if isinstance(val, float) and np.isnan(val):
        return pd.NaT
    val_str = str(val).strip().lower()
    if not val_str or val_str in ("nan", "none", "nat", ""):
        return pd.NaT

    relative_map = [
        (r"^today$", 0),
        (r"^yesterday$", 1),
        (r"^(\d+)\s+days?\s+ago$", "days"),
        (r"^(\d+)\s+hours?\s+ago$", "hours"),
        (r"^(\d+)\s+weeks?\s+ago$", "weeks"),
        (r"^(\d+)\s+months?\s+ago$", "months"),
        (r"^just\s+posted$", 0),
        (r"^be\s+an\s+early\s+applicant$", 0),
        (r"^actively\s+hiring$", 0),
        (r"^30\+\s+days?\s+ago$", 30),
        (r"^over\s+30\s+days?\s+ago$", 30),
        (r"^(\d+)\s+minuten?\s+geleden$", "minutes"),
        (r"^(\d+)\s+uur\s+geleden$", "hours"),
        (r"^(\d+)\s+dagen?\s+geleden$", "days"),
        (r"^(\d+)\s+weken?\s+geleden$", "weeks"),
        (r"^(\d+)\s+maanden?\s+geleden$", "months"),
        (r"^(\d+)\s+minutos?\s+atr[aá]s$", "minutes"),
        (r"^(\d+)\s+horas?\s+atr[aá]s$", "hours"),
        (r"^(\d+)\s+d[ií]as?\s+atr[aá]s$", "days"),
        (r"^(\d+)\s+semanas?\s+atr[aá]s$", "weeks"),
        (r"^(\d+)\s+meses?\s+atr[aá]s$", "months"),
        (r"^il\s+y\s+a\s+(\d+)\s+heure", "hours"),
        (r"^il\s+y\s+a\s+(\d+)\s+jour", "days"),
        (r"^il\s+y\s+a\s+(\d+)\s+semaine", "weeks"),
        (r"^il\s+y\s+a\s+(\d+)\s+mois", "months"),
        (r"^vor\s+(\d+)\s+stunden?$", "hours"),
        (r"^vor\s+(\d+)\s+tagen?$", "days"),
        (r"^vor\s+(\d+)\s+wochen?$", "weeks"),
        (r"^vor\s+(\d+)\s+monaten?$", "months"),
        (r"^(\d+)\s+timmar?\s+sedan$", "hours"),
        (r"^(\d+)\s+dagar?\s+sedan$", "days"),
        (r"^(\d+)\s+veckor?\s+sedan$", "weeks"),
        (r"^(\d+)\s+månader?\s+sedan$", "months"),
        (r"^(\d+)\s+timer?\s+siden$", "hours"),
        (r"^(\d+)\s+dager?\s+siden$", "days"),
        (r"^(\d+)\s+uker?\s+siden$", "weeks"),
        (r"^(\d+)\s+måneder?\s+siden$", "months"),
    ]
    now = pd.Timestamp.now().normalize()
    for pattern, offset in relative_map:
        m = re.search(pattern, val_str)
        if m:
            if isinstance(offset, int):
                return now - pd.Timedelta(days=offset)
            elif offset == "minutes":
                return now - pd.Timedelta(minutes=int(m.group(1)))
            elif offset == "hours":
                return now - pd.Timedelta(hours=int(m.group(1)))
            elif offset == "days":
                return now - pd.Timedelta(days=int(m.group(1)))
            elif offset == "weeks":
                return now - pd.Timedelta(weeks=int(m.group(1)))
            elif offset == "months":
                return now - pd.DateOffset(months=int(m.group(1)))

    explicit_formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d.%m.%Y", "%Y.%m.%d",
        "%d %b %Y", "%d %B %Y",
        "%b %d, %Y", "%B %d, %Y",
        "%b %d %Y", "%B %d %Y",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%d-%b-%Y", "%d %b, %Y",
        "%m/%Y", "%b %Y", "%B %Y",
    ]
    for fmt in explicit_formats:
        try:
            parsed = pd.to_datetime(val_str, format=fmt)
            if pd.notna(parsed):
                return parsed
        except Exception:
            continue

    try:
        parsed = pd.to_datetime(val, infer_datetime_format=True, utc=False)
        if pd.notna(parsed):
            return parsed
    except Exception:
        pass

    logger.warning(f"  parse_date unresolved: repr={repr(val)}")
    return pd.NaT


def normalize_country(raw: str) -> str:
    if not raw or raw.strip().lower() in ("nan", "none", "", "unknown"):
        return "Unknown"

    s = raw.strip()
    s_lower = s.lower()

    if s_lower in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[s_lower]

    s_clean = re.sub(r"\s+(metropolitan area|metro area|greater|area)\b.*$", "", s_lower).strip()
    if s_clean in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[s_clean]

    if re.search(r"metropolitan area|metro area", s_lower):
        for city_kw, country in _METRO_AREA_COUNTRY:
            if city_kw in s_lower:
                return country

    if re.fullmatch(r"[A-Z]{2}", s) and s in _US_STATE_ABBR:
        return "United States"

    if re.fullmatch(r"[A-Z]{2}", s):
        return "Unknown"

    for city in _US_CITIES:
        if city in s_lower:
            return "United States"

    for alias, country in _COUNTRY_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", s_lower):
            return country

    if len(s) > 2 and s[0].isupper():
        return s
    return "Unknown"


def parse_location(loc) -> tuple:
    if loc is None or (isinstance(loc, float) and np.isnan(loc)):
        return None, "Unknown"
    loc_str = str(loc).strip()
    if not loc_str or loc_str.lower() in ("nan", "none", ""):
        return None, "Unknown"

    parts = [p.strip() for p in re.split(r"[,\|]+", loc_str) if p.strip()]

    country_raw = None
    city = None

    if len(parts) == 0:
        country_raw = loc_str
    elif len(parts) == 1:
        country_raw = parts[0]
        city = None
    elif len(parts) == 2:
        city = parts[0]
        country_raw = parts[1]
    else:
        city = parts[0]
        country_raw = parts[-1]

    country = normalize_country(country_raw or "")

    if country == "Unknown" and city:
        country = normalize_country(city)
        if country != "Unknown":
            city = None

    return city, country


def infer_job_level(title) -> str:
    if not title or (isinstance(title, float) and np.isnan(title)):
        return "Unknown"
    t = str(title).lower()
    for level, keywords in JOB_LEVEL_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return level
    return "Unknown"


def _kw_match(kw: str, text: str) -> bool:
    if " " in kw:
        return kw in text
    return bool(re.search(r"\b" + re.escape(kw) + r"\b", text))


def infer_job_category(title, title_original=None) -> str:
    if not title or (isinstance(title, float) and np.isnan(title)):
        return "Other"
    t = str(title).lower()
    for category, keywords in JOB_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if _kw_match(kw, t):
                return category
    words = set(re.findall(r"\b\w+\b", t))
    for category, partial_kws in JOB_CATEGORY_PARTIAL_KEYWORDS.items():
        for kw in partial_kws:
            if kw in words:
                return category
    raw = str(title_original).lower() if title_original and not (isinstance(title_original, float) and np.isnan(title_original)) else t
    for category, keywords in JOB_CATEGORY_KEYWORDS_MULTILINGUAL.items():
        for kw in keywords:
            if kw in raw:
                return category
    return "Other"


def normalize_salary_col(series: pd.Series) -> pd.Series:
    def _parse(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (int, float)):
            return float(v) if np.isfinite(v) else np.nan
        s = str(v).replace(",", "").replace("$", "").replace("£", "").replace("€", "").strip()
        m = re.search(r"[\d]+(?:\.\d+)?", s)
        if m:
            val = float(m.group())
            if "k" in s.lower():
                val *= 1000
            return val
        return np.nan
    return series.apply(_parse)


def normalize_pay_period(df: pd.DataFrame) -> pd.DataFrame:
    period_col = None
    for candidate in ["pay_period", "salary_period", "compensation_type", "pay_type"]:
        if candidate in df.columns:
            period_col = candidate
            break

    if period_col is None:
        min_vals = df["salary_min"].dropna()
        max_vals = df["salary_max"].dropna()
        all_vals = pd.concat([min_vals, max_vals])
        if len(all_vals) > 0:
            likely_hourly = (all_vals < HOURLY_SALARY_THRESHOLD).mean()
            if likely_hourly > 0.5:
                logger.warning(f"  No pay_period column found but >50% values look hourly — applying x2080 to all")
                df["salary_min"] = df["salary_min"] * 2080
                df["salary_max"] = df["salary_max"] * 2080
                df["salary_period_norm"] = "hourly_inferred"
            else:
                df["salary_period_norm"] = "yearly_assumed"
        return df

    def _multiplier(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 1
        return PAY_PERIOD_MULTIPLIER.get(str(val).strip().lower(), 1)

    multipliers = df[period_col].apply(_multiplier)
    df["salary_period_norm"] = df[period_col].apply(
        lambda v: str(v).strip().lower() if pd.notna(v) else "unknown"
    )

    needs_conversion = multipliers != 1
    converted_count = needs_conversion.sum()
    if converted_count > 0:
        logger.warning(f"  Pay period normalization: converting {converted_count} rows to yearly")

    df["salary_min"] = df["salary_min"].where(~needs_conversion, df["salary_min"] * multipliers)
    df["salary_max"] = df["salary_max"].where(~needs_conversion, df["salary_max"] * multipliers)
    return df


def normalize_salary(df: pd.DataFrame) -> pd.DataFrame:
    df["salary_min"] = normalize_salary_col(df.get("salary_min", pd.Series(dtype=float)))
    df["salary_max"] = normalize_salary_col(df.get("salary_max", pd.Series(dtype=float)))

    df["salary_min"] = df["salary_min"].where(df["salary_min"] > 0)
    df["salary_max"] = df["salary_max"].where(df["salary_max"] > 0)

    df = normalize_pay_period(df)

    swap_mask = (df["salary_min"].notna() & df["salary_max"].notna() &
                 (df["salary_min"] > df["salary_max"]))
    df.loc[swap_mask, ["salary_min", "salary_max"]] = (
        df.loc[swap_mask, ["salary_max", "salary_min"]].values
    )

    df["salary_max"] = df["salary_max"].where(
        df["salary_max"] >= df["salary_min"].fillna(0))

    level_col = df.get("job_level", pd.Series(["Unknown"] * len(df)))
    total_nulled = 0
    for level, (lo, hi) in SALARY_BOUNDS_BY_LEVEL.items():
        mask = level_col == level
        before_min = df.loc[mask, "salary_min"].notna().sum()
        before_max = df.loc[mask, "salary_max"].notna().sum()
        df.loc[mask, "salary_min"] = df.loc[mask, "salary_min"].where(
            df.loc[mask, "salary_min"].between(lo, hi))
        df.loc[mask, "salary_max"] = df.loc[mask, "salary_max"].where(
            df.loc[mask, "salary_max"].between(lo, hi))
        nulled = (before_min - df.loc[mask, "salary_min"].notna().sum()) + \
                 (before_max - df.loc[mask, "salary_max"].notna().sum())
        if nulled > 0:
            logger.warning(f"  Salary bounds [{level}] nulled {nulled} values outside ({lo}, {hi})")
        total_nulled += nulled

    if total_nulled > 0:
        logger.warning(f"  Salary bounds total nulled: {total_nulled} values")

    df["has_salary"] = (df["salary_min"].notna() | df["salary_max"].notna())
    return df


def normalize_remote(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if np.isnan(val) if isinstance(val, float) else False:
            return False
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ["true", "yes", "1", "remote",
                                        "work from home", "wfh", "fully remote"]
    return False


def normalize_platform(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "LinkedIn"
    mapping = {
        "linkedin": "LinkedIn",
        "indeed": "Indeed",
        "glassdoor": "Glassdoor",
        "zip_recruiter": "ZipRecruiter",
        "ziprecruiter": "ZipRecruiter",
        "google": "Google Jobs",
        "google jobs": "Google Jobs",
        "google_jobs": "Google Jobs",
        "kaggle": "LinkedIn",
        "kaggle_2024": "LinkedIn",
        "kaggle_linkedin": "LinkedIn",
        "kaggle linkedin": "LinkedIn",
    }
    normalized = mapping.get(str(val).lower().strip())
    if normalized:
        return normalized
    known = {"linkedin", "indeed", "glassdoor", "ziprecruiter", "google jobs"}
    candidate = str(val).strip().title()
    if candidate.lower() in {k.lower() for k in known}:
        return candidate
    return "LinkedIn"


def _make_source_hash(row: pd.Series) -> str:
    company = str(row.get("company_clean") or "")
    title = str(row.get("title_clean") or "")
    date = str(row.get("date_parsed") or "")
    location = str(row.get("loc_country") or row.get("location") or "")
    raw = f"{company}|{title}|{date}|{location}"
    return hashlib.md5(raw.encode()).hexdigest()


def _tag_languages(df: pd.DataFrame) -> pd.DataFrame:
    logger.warning("  Detecting description languages...")
    df["description_lang"] = df["description_clean"].apply(detect_language)
    lang_counts = df["description_lang"].value_counts().to_dict()
    logger.warning(f"  Language distribution: {lang_counts}")
    non_english = df[~df["description_lang"].isin(["en", "unknown"])]["description_lang"].value_counts().to_dict()
    if non_english:
        logger.warning(f"  Non-English rows (kept, NER still runs): {non_english}")
    return df


def preprocess(df: pd.DataFrame, source_label: str = "unknown") -> pd.DataFrame:
    df = df.copy()
    logger.warning(f"Preprocessing {len(df)} rows, source={source_label}")

    title_col = df.get("title", pd.Series(dtype=str))
    df["title_clean"] = title_col.apply(clean_text)
    df["description_clean"] = df.get("description", pd.Series(dtype=str)).apply(clean_text)
    df["description_ner"] = df.get("description", pd.Series(dtype=str)).apply(clean_text_ner)
    df["company_clean"] = df.get("company", pd.Series(dtype=str)).apply(
        lambda x: clean_text(x).title() if clean_text(x) else "Unknown"
    )

    df["date_parsed"] = df.get("date_posted", pd.Series(dtype=object)).apply(parse_date)
    df["date_parsed"] = pd.to_datetime(df["date_parsed"], utc=False, errors="coerce")
    df["date_parsed"] = df["date_parsed"].apply(
        lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, "tzinfo") and x.tzinfo is not None else x
    )

    nat_mask = df["date_parsed"].isna()
    nat_count = nat_mask.sum()
    if nat_count > 0:
        scraping_date = pd.Timestamp.now().normalize()
        logger.warning(f"  Filling {nat_count} unresolvable dates with scraping date: {scraping_date.date()}")
        df.loc[nat_mask, "date_parsed"] = scraping_date

    before = len(df)
    df = df[df["date_parsed"] >= pd.Timestamp("2023-01-01")].copy()
    after_old = len(df)
    df = df[df["date_parsed"] <= pd.Timestamp.now() + pd.Timedelta(days=1)].copy()
    after_future = len(df)
    logger.warning(
        f"  Date filter: {before}"
        f" → {after_old} (dropped {before - after_old} pre-2023)"
        f" → {after_future} (dropped {after_old - after_future} future)"
    )

    df["date_parsed"] = df["date_parsed"].dt.normalize()

    loc_parsed = df.get("location", pd.Series(dtype=str)).apply(parse_location)
    df["loc_city"] = loc_parsed.apply(lambda x: x[0])
    df["loc_country"] = loc_parsed.apply(lambda x: x[1])
    df["loc_country"] = df["loc_country"].apply(lambda c: c if c in REGION_MAP else "Unknown")
    if "search_location" in df.columns:
        unknown_mask = df["loc_country"] == "Unknown"
        if unknown_mask.any():
            df.loc[unknown_mask, "loc_country"] = df.loc[unknown_mask, "search_location"].apply(
                lambda x: normalize_country(str(x))
                if pd.notna(x) and str(x).strip().lower() not in ("nan", "none", "")
                else "Unknown"
            )
            logger.warning(f"  Filled {unknown_mask.sum()} unknown countries from search_location")
    df["global_region"] = df["loc_country"].map(REGION_MAP).fillna("Other")

    df["job_level"] = df["title_clean"].apply(infer_job_level)
    df["job_category"] = df.apply(
        lambda row: infer_job_category(row["title_clean"], row.get("title", "")),
        axis=1,
    )

    df = normalize_salary(df)
    df["is_remote"] = df.get("is_remote", pd.Series(dtype=object)).apply(normalize_remote)
    df["platform_norm"] = df.get("platform", pd.Series(dtype=str)).apply(normalize_platform)
    df["source_label"] = source_label

    df["source_hash"] = df.apply(_make_source_hash, axis=1)

    before = len(df)
    df = df.drop_duplicates(subset=["source_hash"])
    logger.warning(f"  Internal dedup: {before} → {len(df)} rows (removed {before - len(df)})")

    before = len(df)
    df = df[df["title_clean"].str.len() > 1].copy()
    after_title = len(df)
    df = df[df["description_clean"].str.len() >= 10].copy()
    after_desc = len(df)
    logger.warning(
        f"  Quality filter: {before} → {after_title} (dropped {before - after_title} short title)"
        f" → {after_desc} (dropped {after_title - after_desc} short desc)"
    )

    df = _tag_languages(df)

    logger.warning(f"  Preprocessing done: {len(df)} rows remain")
    return df.reset_index(drop=True)


def preprocess_file(input_path: str, source_label: str = "unknown") -> Path:
    p = Path(input_path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".csv", ".tsv"):
        df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    df = preprocess(df, source_label=source_label)

    out_path = DATA_PROCESSED_DIR / (p.stem + "_preprocessed.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.warning(f"Preprocessed output: {len(df)} rows → {out_path}")
    return out_path