import pandas as pd

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

def add_narration_column(df):
    """
    Add narration column to dataframe
    """
    print(f"🎬 Generating narration for {len(df)} movies...")
    # df['release_date'] = df['release_date'].dt.date

    df['narration'] = df.apply(generate_natural_movie_narration, axis=1)
    
    # Check empty narrations
    empty_count = df['narration'].apply(lambda x: len(x) == 0).sum()
    print(f"✅ Narration generated: {len(df) - empty_count} movies")
    if empty_count > 0:
        print(f"⚠️ {empty_count} movies have empty narration (missing title & description)")
    return df

def main():
    train_set = pd.read_csv("dataset/q3/260826/260827_final_api_train_movies.csv")
    train_set = add_narration_column(train_set)
    col_order = [col for col in train_set.columns if col!='explanatory_factors'] + ['explanatory_factors']
    train_set_sorted = train_set[col_order].copy()
    print(train_set_sorted.info())
    train_set_sorted.to_csv("dataset/q3/260826/260827_final_api_train_movies_with_narration.csv",index=False)
    
if __name__ == "__main__":
    main()