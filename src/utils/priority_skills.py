# Deterministic "high-priority" skills set for Critical Issues counting.
# You can extend this list or read from a CSV if needed.
HIGH_PRIORITY_SKILLS = {
    # Core languages/stacks
    "python","java","c++","c#",".net","node.js","javascript","typescript","go","rust",
    "sql","mysql","postgresql","oracle","mongodb","spark","hadoop",
    # Cloud
    "aws","azure","gcp","google cloud","kubernetes","docker","terraform","ansible",
    # Data/ML
    "pandas","numpy","scikit-learn","tensorflow","pytorch","mlops",
    # DevOps/infra
    "linux","git","ci/cd","jenkins","kafka","elastic","elasticsearch","snowflake",
    # Analytics/bi
    "power bi","tableau","looker","excel",
    # Security
    "siem","soc","iam","okta","splunk"
}

def get_priority_skills():
    return HIGH_PRIORITY_SKILLS
