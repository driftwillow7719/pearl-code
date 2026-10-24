import re
import pandas as pd
import numpy as np

def clean_text(text):
    """
    Clean text from weird symbols, latin characters, etc.
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove special characters but keep letters, numbers, and basic punctuation
    text = re.sub(r'[^\w\s\-\.\,\'\"\(\)\!\?]', ' ', text) 
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Fix apostrophe encoding issues
    text = text.replace(" 39 s", "'s")
    text = text.replace(" 39 ", "' ")
    text = text.replace("&apos;", "'")
    text = text.replace("&#39;", "'")
    text = text.replace("quot ", '"')
    text = text.replace(" quot", '"')

    return text

def remove_duplicate_names(text, separator=','):
    """
    Remove duplicate names in a comma-separated string.
    Example: "A, B, A, C" -> "A, B, C"
    """
    if pd.isna(text) or text == '':
        return text
    
    # Split by comma, strip whitespace, and remove duplicates while preserving order
    items = [item.strip() for item in str(text).split(separator)]
    seen = set()
    unique_items = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            unique_items.append(item)
    
    return separator.join(unique_items)

def clean_plot(text, max_chars=500):
    """
    Extract only the story/plot, remove names and metadata
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    # Take only synopsis part (after "Synopsis:")
    if 'Synopsis:' in text:
        text = text.split('Synopsis:')[-1].strip()
    
    # Remove parenthesized names (actor names)
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove common patterns: "Name: Character" or "Name as Character"
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\s+(?:as|is)\s+\w+', '[ACTOR]', text)
    
    # Remove director/writer mentions if they appear in plot
    text = re.sub(r'(Directed by|Written by|Produced by|Starring)\s+[A-Z][a-z]+(?: [A-Z][a-z]+)*', '', text)
    
    # Clean extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate to max_chars
    if len(text) > max_chars:
        text = text[:max_chars] + '...'
    
    return text

def clean_cast(text, max_actors=5):
    """Clean and truncate cast list"""
    if not text or pd.isna(text):
        return ""
    
    # Use existing clean_text function
    text = clean_text(text)
    
    # Remove duplicates using existing function
    text = remove_duplicate_names(text, separator=',')
    
    # Truncate to top N actors
    actors = [a.strip() for a in text.split(',') if a.strip()]
    actors = actors[:max_actors]
    
    return ', '.join(actors)
        
def clean_column_duplicates(df, column_name, separator=','):
    """
    Apply duplicate removal to a specific column
    """
    df[column_name] = df[column_name].apply(lambda x: remove_duplicate_names(x, separator))
    return df

def clean_column_text(df, column_name):
    """
    Apply text cleaning to a specific column
    """
    df[column_name] = df[column_name].apply(clean_text)
    return df

def clean_movie_data(df):
    """
    Master function to clean all specified columns
    """
    df_clean = df.copy()
    
    for col in df.columns:
        df_clean = clean_column_text(df_clean, col)
        # First remove duplicates if it's a list-like column
        if col in ['top5_actors', 'writers', 'producers', 'music']:
            df_clean = clean_column_duplicates(df_clean, col, separator=',')
    # Special handling for release_date
    if 'release_date' in df_clean.columns:
        df_clean['release_date'] = pd.to_datetime(df_clean['release_date'], errors='coerce')
    return df_clean

# def generate_movie_narration(row):
#     """
#     Generate natural language narration from movie metadata columns
#     """
#     parts = []
    
#     # 1. Title
#     title = row.get('movie_name', '')
#     if pd.notna(title) and str(title).strip():
#         parts.append(f"'{str(title).strip()}'")
    
#     # 2. Genre
#     genre = row.get('genre', '')
#     if pd.notna(genre) and str(genre).strip():
#         parts.append(f"is a {str(genre).strip()} movie")
    
#     # 3. Language
#     language = row.get('language', '')
#     if pd.notna(language) and str(language).strip():
#         parts.append(f"in {str(language).strip()}")
    
#     # 4. Release date (extract year)
#     release_date = row.get('release_date', '')
#     if pd.notna(release_date) and str(release_date).strip() != 'NaT':
#         try:
#             if isinstance(release_date, str):
#                 year = release_date.split('-')[0]
#             else:
#                 year = release_date.year
#             parts.append(f"released in {year}")
#         except:
#             parts.append(f"released in {str(release_date)}")
    
#     # 5. Director
#     director = row.get('director', '')
#     if pd.notna(director) and str(director).strip():
#         parts.append(f"directed by {str(director).strip()}")
    
#     # 6. Writers
#     writers = row.get('writers', '')
#     if pd.notna(writers) and str(writers).strip():
#         parts.append(f"written by {str(writers).strip()}")
    
#     # 7. Top 5 Actors
#     actors = row.get('top5_actors', '')
#     if pd.notna(actors) and str(actors).strip():
#         parts.append(f"starring {str(actors).strip()}")
    
#     # 8. Producers
#     producers = row.get('producers', '')
#     if pd.notna(producers) and str(producers).strip():
#         parts.append(f"produced by {str(producers).strip()}")
    
#     # 9. Music
#     music = row.get('music', '')
#     if pd.notna(music) and str(music).strip():
#         parts.append(f"with music by {str(music).strip()}")
    
#     # 10. Description (as synopsis)
#     description = row.get('description', '')
#     if pd.notna(description) and str(description).strip():
#         parts.append(f"\nSynopsis: {str(description).strip()}")
    
#     # Combine all parts
#     if not parts:
#         return ''
    
#     narration = parts[0]
#     for part in parts[1:]:
#         if part.startswith('\n'):
#             narration += part
#         else:
#             narration += " " + part
    
#     return narration

def generate_natural_movie_narration(row):
    """
    Generate natural language narration from movie metadata columns
    """
    def get(col):
        val = row.get(col, '')
        return str(val).strip() if pd.notna(val) and str(val).strip() not in ('', 'NaT') else None

    title    = get('movie_name')
    genre    = get('genre')
    language = get('language')
    director = get('director')
    writers  = get('writers')
    actors   = get('top5_actors')
    producers= get('producers')
    music    = get('music')
    synopsis = get('description')

    # Extract full date
    date_str = None
    release_date = row.get('release_date', '')
    if pd.notna(release_date) and str(release_date).strip() not in ('', 'NaT'):
        try:
            date_str = release_date.strftime('%B %d, %Y')   # → "July 16, 2010"
        except:
            pass

    # --- Build metadata sentence ---
    # Opening: "Inception (2010) is an English-language Action film"
    opening = title or "This film"
    if date_str:
        opening += f" ({date_str})"
    if genre or language:
        lang_str  = f"{language}-language " if language else ""
        genre_str = f"{genre} film" if genre else "film"
        opening  += f" is a {lang_str}{genre_str}"
    opening += "."

    # Crew line: "Directed by X, written by Y, and produced by Z."
    crew_parts = []
    if director:  crew_parts.append(f"directed by {director}")
    if writers:   crew_parts.append(f"written by {writers}")
    if producers: crew_parts.append(f"produced by {producers}")
    crew_line = (crew_parts[0].capitalize() + (
        ", " + ", ".join(crew_parts[1:-1]) + (f", and {crew_parts[-1]}" if len(crew_parts) > 1 else "")
        if len(crew_parts) > 1 else ""
    ) + ".") if crew_parts else None

    # Cast/music line: "Starring A, B, and C, with music by D."
    cast_line = None
    if actors or music:
        cast_line = (f"Starring {actors}" if actors else "") + \
                    (f", with music by {music}" if music else "") + "."
        cast_line = cast_line.lstrip(", ")

    # Synopsis
    synopsis_line = f"Synopsis: {synopsis}" if synopsis else None

    return "\n".join(filter(None, [opening, crew_line, cast_line, synopsis_line]))
