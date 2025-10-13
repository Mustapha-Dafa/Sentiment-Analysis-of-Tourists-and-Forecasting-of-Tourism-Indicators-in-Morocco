"""
Google Trends Scraper via SerpAPI avec SAUVEGARDE PROGRESSIVE
Scrape les tendances de recherche pour une liste de mots-clés
Période: 01/2010 - 06/2025 | Région: Worldwide
✅ Sauvegarde après CHAQUE keyword (protection contre timeouts API)
✅ Reprise automatique si interruption
"""

import pandas as pd
import requests
import time
from pathlib import Path
from datetime import datetime
import json

# ========= CONFIGURATION =========
SERPAPI_KEY = "your serapi key"  # ⚠️ Remplacez par votre clé SerpAPI
INPUT_FILE = "src/Google Trend/keywords_clean.csv"  # Fichier avec vos keywords
OUTPUT_DIR = Path("src/Google Trend/google_trends_data_test")

# Paramètres de scraping
START_DATE = "2010-01-01"
END_DATE = "2025-06-30"
REGION = ""  # Vide = Worldwide, ou "US", "MA", etc.
DELAY = 2  # Secondes entre chaque requête
MAX_RETRIES = 3  # Tentatives max par keyword en cas d'erreur

# Fichiers de tracking
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
CONSOLIDATED_FILE = OUTPUT_DIR / "all_trends_consolidated_test.csv"

# ========= INITIALISATION =========
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_keywords(filepath):
    """Charge la liste des mots-clés"""
    df = pd.read_csv(filepath)
    col = "term" if "term" in df.columns else df.columns[0]
    keywords = df[col].dropna().unique().tolist()
    print(f"✓ {len(keywords)} mots-clés chargés depuis {filepath}")
    return keywords

def load_progress():
    """Charge l'état de progression (pour reprise)"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
        print(f"✓ Progression chargée: {len(progress['completed'])} déjà scrapés")
        return progress
    return {"completed": [], "failed": [], "last_updated": None}

def save_progress(progress):
    """Sauvegarde l'état de progression"""
    progress["last_updated"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

def scrape_google_trends(keyword, api_key):
    """
    Scrape Google Trends pour un mot-clé via SerpAPI
    Doc: https://serpapi.com/google-trends-api
    """
    params = {
        "engine": "google_trends",
        "q": keyword,
        "date": f"{START_DATE} {END_DATE}",
        "data_type": "TIMESERIES",
        "api_key": api_key
    }
    
    if REGION:
        params["geo"] = REGION
    
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Extraire les données de tendance
        if "interest_over_time" in data:
            timeline = data["interest_over_time"].get("timeline_data", [])
            return {
                "keyword": keyword,
                "status": "success",
                "data_points": len(timeline),
                "timeline": timeline,
                "raw_response": data
            }
        else:
            return {
                "keyword": keyword,
                "status": "no_data",
                "data_points": 0,
                "timeline": [],
                "error": "Aucune donnée disponible"
            }
    
    except requests.exceptions.Timeout:
        return {"keyword": keyword, "status": "timeout", "error": "Timeout API"}
    
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            return {"keyword": keyword, "status": "rate_limit", "error": "Rate limit atteint"}
        return {"keyword": keyword, "status": "error", "error": f"HTTP {response.status_code}"}
    
    except Exception as e:
        return {"keyword": keyword, "status": "error", "error": str(e)}

def save_keyword_data(result, keyword):
    """
    SAUVEGARDE IMMÉDIATE après chaque keyword
    Crée 3 fichiers:
    1. Timeline CSV individuel
    2. JSON complet individuel
    3. Ajout au fichier consolidé
    """
    safe_name = keyword.replace(' ', '_').replace('/', '-')[:50]
    
    # 1. Timeline CSV individuel
    if result["timeline"]:
        timeline_df = pd.DataFrame(result["timeline"])
        timeline_df["keyword"] = keyword
        timeline_df["scraped_at"] = datetime.now().isoformat()
        
        # timeline_file = OUTPUT_DIR / f"{safe_name}_timeline.csv"
        # timeline_df.to_csv(timeline_file, index=False)
        
        # 2. Ajout au fichier CONSOLIDÉ (pour analyse globale)
        if CONSOLIDATED_FILE.exists():
            consolidated = pd.read_csv(CONSOLIDATED_FILE)
            consolidated = pd.concat([consolidated, timeline_df], ignore_index=True)
        else:
            consolidated = timeline_df
        consolidated.to_csv(CONSOLIDATED_FILE, index=False)
    
    # 3. JSON complet (backup raw)
    # json_file = OUTPUT_DIR / f"{safe_name}_raw.json"
    # result_copy = result.copy()
    # result_copy["scraped_at"] = datetime.now().isoformat()
    # with open(json_file, "w", encoding="utf-8") as f:
    #     json.dump(result_copy, f, indent=2, ensure_ascii=False)

def main():
    """Fonction principale avec SAUVEGARDE PROGRESSIVE"""
    print("\n" + "="*70)
    print("GOOGLE TRENDS SCRAPER - SERPAPI (SAUVEGARDE PROGRESSIVE)")
    print("="*70)
    
    # Vérifier la clé API
    if SERPAPI_KEY == "VOTRE_CLE_API":
        print("\n⚠️  ERREUR: Veuillez configurer votre clé SerpAPI dans le script")
        print("   Obtenez votre clé sur: https://serpapi.com/")
        return
    
    # Charger les mots-clés
    try:
        all_keywords = load_keywords(INPUT_FILE)
    except FileNotFoundError:
        print(f"\n⚠️  ERREUR: Fichier {INPUT_FILE} introuvable")
        return
    
    # Charger la progression (pour reprise)
    progress = load_progress()
    remaining_keywords = [k for k in all_keywords if k not in progress["completed"]]
    
    print(f"\n📊 Configuration:")
    print(f"   • Période: {START_DATE} → {END_DATE}")
    print(f"   • Région: {REGION if REGION else 'Worldwide'}")
    print(f"   • Total mots-clés: {len(all_keywords)}")
    print(f"   • Déjà scrapés: {len(progress['completed'])}")
    print(f"   • Restants: {len(remaining_keywords)}")
    print(f"   • Délai entre requêtes: {DELAY}s")
    
    if not remaining_keywords:
        print("\n✅ Tous les mots-clés ont déjà été scrapés!")
        print(f"📁 Données dans: {OUTPUT_DIR}/")
        return
    
    # Scraping avec SAUVEGARDE APRÈS CHAQUE KEYWORD
    results_summary = []
    start_time = datetime.now()
    
    print(f"\n🚀 Démarrage du scraping...\n")
    
    for i, keyword in enumerate(remaining_keywords, 1):
        total_progress = len(progress["completed"]) + i
        print(f"[{total_progress}/{len(all_keywords)}] Scraping: {keyword[:45]}...", end=" ")
        
        # Tentatives avec retry
        result = None
        for attempt in range(MAX_RETRIES):
            result = scrape_google_trends(keyword, SERPAPI_KEY)
            
            if result["status"] == "success":
                # ✅ SAUVEGARDE IMMÉDIATE
                save_keyword_data(result, keyword)
                progress["completed"].append(keyword)
                save_progress(progress)
                print(f"✓ ({result['data_points']} pts) [SAVED]")
                break
            
            elif result["status"] == "no_data":
                progress["completed"].append(keyword)
                progress["failed"].append({"keyword": keyword, "reason": "no_data"})
                save_progress(progress)
                print("⚠️  Aucune donnée [SAVED]")
                break
            
            elif result["status"] == "rate_limit":
                print(f"⛔ Rate limit - Pause 60s (tentative {attempt+1}/{MAX_RETRIES})...")
                time.sleep(60)
                if attempt == MAX_RETRIES - 1:
                    progress["failed"].append({"keyword": keyword, "reason": "rate_limit"})
                    save_progress(progress)
            
            elif result["status"] == "timeout":
                print(f"⏱️  Timeout - Retry {attempt+1}/{MAX_RETRIES}...", end=" ")
                time.sleep(5)
                if attempt == MAX_RETRIES - 1:
                    progress["failed"].append({"keyword": keyword, "reason": "timeout"})
                    save_progress(progress)
                    print("❌ Échec après retries")
            
            else:
                print(f"❌ Erreur: {result.get('error', 'Unknown')}")
                progress["failed"].append({"keyword": keyword, "reason": result.get("error")})
                save_progress(progress)
                break
        
        # Log summary
        results_summary.append({
            "keyword": keyword,
            "status": result["status"] if result else "unknown",
            "data_points": result.get("data_points", 0) if result else 0,
            "scraped_at": datetime.now().isoformat()
        })
        
        # Délai entre requêtes
        if i < len(remaining_keywords):
            time.sleep(DELAY)
    
    # Rapport final
    duration = (datetime.now() - start_time).total_seconds()
    summary_df = pd.DataFrame(results_summary)
    summary_file = OUTPUT_DIR / f"scraping_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print("\n" + "="*70)
    print("RAPPORT FINAL")
    print("="*70)
    print(f"⏱️  Durée session: {duration/60:.1f} minutes")
    print(f"✓ Total scrapés: {len(progress['completed'])}/{len(all_keywords)}")
    print(f"❌ Échecs: {len(progress['failed'])}")
    print(f"\n📁 Fichiers sauvegardés dans: {OUTPUT_DIR}/")
    print(f"   • {len(progress['completed'])} × _timeline.csv (données individuelles)")
    print(f"   • all_trends_consolidated.csv (TOUTES les données)")
    print(f"   • progress.json (état de progression)")
    print(f"   • scraping_log_*.csv (logs de cette session)")
    
    if len(remaining_keywords) > len(progress['completed']):
        print(f"\n⚠️  Il reste {len(all_keywords) - len(progress['completed'])} mots-clés")
        print("   Relancez le script pour continuer!")

if __name__ == "__main__":
    main()