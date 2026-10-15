"""
analytics.py — Data processing and backend metrics for AI Resume & Job Market Analytics dashboard.
"""

from collections import Counter
import itertools
from pathlib import Path
import pandas as pd


def load_data(filepath: str = "AI_Resume_Job_Market_Analytics_500 (1).csv") -> pd.DataFrame:
    """Load and preprocess the job market dataset."""
    path = Path(filepath)
    if not path.exists():
        # Fallback if filename differs slightly or path is relative
        files = list(Path(".").glob("*.csv"))
        if files:
            path = files[0]
        else:
            raise FileNotFoundError(f"Dataset CSV file not found at {filepath}")

    df = pd.read_csv(path)

    # Compute average salary mid-point
    df["Salary_Avg_LPA"] = ((df["Salary_Min_LPA"] + df["Salary_Max_LPA"]) / 2).round(2)

    return df


def total_job_postings(df: pd.DataFrame) -> int:
    """Return total number of job postings."""
    return len(df)


def salary_stats(df: pd.DataFrame) -> dict:
    """Return key salary metrics (LPA)."""
    return {
        "Overall Min (LPA)": round(df["Salary_Min_LPA"].min(), 2),
        "Overall Max (LPA)": round(df["Salary_Max_LPA"].max(), 2),
        "Average Mid-Point (LPA)": round(df["Salary_Avg_LPA"].mean(), 2),
        "Median Mid-Point (LPA)": round(df["Salary_Avg_LPA"].median(), 2),
    }


def experience_stats(df: pd.DataFrame) -> dict:
    """Return experience statistics in years."""
    return {
        "Average (yrs)": round(df["Experience_Years"].mean(), 1),
        "Minimum (yrs)": int(df["Experience_Years"].min()),
        "Maximum (yrs)": int(df["Experience_Years"].max()),
        "Median (yrs)": float(df["Experience_Years"].median()),
    }


def postings_by_role(df: pd.DataFrame) -> pd.DataFrame:
    """Return number of job postings broken down by job role."""
    role_counts = (
        df["Job_Title"]
        .value_counts()
        .reset_index()
    )
    role_counts.columns = ["Job Title", "Postings"]
    return role_counts


def work_mode_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return breakdown by work mode (Onsite, Remote, Hybrid)."""
    wm_counts = (
        df["Work_Mode"]
        .value_counts()
        .reset_index()
    )
    wm_counts.columns = ["Work Mode", "Count"]
    return wm_counts


def jobs_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """Return job postings per location/city."""
    loc_counts = (
        df["Location"]
        .value_counts()
        .reset_index()
    )
    loc_counts.columns = ["Location", "Postings"]
    return loc_counts


def education_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return frequency distribution of education requirements."""
    edu_counts = (
        df["Education"]
        .value_counts()
        .reset_index()
    )
    edu_counts.columns = ["Education", "Count"]
    return edu_counts


def avg_salary_by_role(df: pd.DataFrame) -> pd.DataFrame:
    """Return average mid-point salary per job role."""
    sal = (
        df.groupby("Job_Title")["Salary_Avg_LPA"]
        .mean()
        .round(2)
        .reset_index()
    )
    sal.columns = ["Job Title", "Avg Salary (LPA)"]
    return sal.sort_values(by="Avg Salary (LPA)", ascending=False)


def avg_experience_by_role(df: pd.DataFrame) -> pd.DataFrame:
    """Return average experience requirement per job role."""
    exp = (
        df.groupby("Job_Title")["Experience_Years"]
        .mean()
        .round(1)
        .reset_index()
    )
    exp.columns = ["Job Title", "Avg Experience (yrs)"]
    return exp.sort_values(by="Avg Experience (yrs)", ascending=False)


def technical_skill_frequency(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Extract and rank individual technical skills from comma-separated column."""
    skills = []
    for row in df["Required_Skills"].dropna():
        skills.extend([s.strip() for s in str(row).split(",") if s.strip()])
    counts = Counter(skills).most_common(top_n)
    return pd.DataFrame(counts, columns=["Skill", "Frequency"])


def soft_skill_frequency(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Extract and rank soft skills from comma-separated column."""
    skills = []
    for row in df["Soft_Skills"].dropna():
        skills.extend([s.strip() for s in str(row).split(",") if s.strip()])
    counts = Counter(skills).most_common(top_n)
    return pd.DataFrame(counts, columns=["Soft Skill", "Frequency"])


def skill_combinations(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Calculate co-occurrences of technical skill pairs."""
    pair_list = []
    for row in df["Required_Skills"].dropna():
        skills = sorted(list(set([s.strip() for s in str(row).split(",") if s.strip()])))
        pair_list.extend(list(itertools.combinations(skills, 2)))

    counts = Counter(pair_list).most_common(top_n)
    data = [
        {"Skill A": pair[0], "Skill B": pair[1], "Co-occurrences": freq}
        for pair, freq in counts
    ]
    return pd.DataFrame(data)


def experience_salary_scatter(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe prepared for scatter plotting experience vs salary."""
    return df.copy()
