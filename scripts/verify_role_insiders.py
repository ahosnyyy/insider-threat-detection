import pandas as pd
from pathlib import Path

def main():
    raw_dir = Path("data/raw")
    answers_dir = raw_dir / "answers"
    ldap_dir = raw_dir / "LDAP"
    
    # 1. Load Insiders
    insiders_path = answers_dir / "insiders.csv"
    if not insiders_path.exists():
        print(f"Error: {insiders_path} not found")
        return
        
    print(f"Loading insiders from {insiders_path}...")
    insiders_df = pd.read_csv(insiders_path)
    # user column might be 'user' or 'user_id'
    user_col = 'user' if 'user' in insiders_df.columns else 'user_id'
    insider_users = set(insiders_df[user_col].unique())
    print(f"Found {len(insider_users)} unique insider users.")
    
    # 2. Check 'AdministrativeStaff' role in LDAP files
    target_role = 'AdministrativeStaff'
    users_with_role = set()
    
    print(f"\nScanning LDAP files for role '{target_role}'...")
    ldap_files = sorted(ldap_dir.glob("*.csv"))
    
    for f in ldap_files:
        try:
            df = pd.read_csv(f)
            # Columns usually: employee_name, user_id, email, role, etc.
            if 'role' in df.columns and 'user_id' in df.columns:
                role_users = df[df['role'] == target_role]['user_id'].tolist()
                users_with_role.update(role_users)
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            
    print(f"Found {len(users_with_role)} users who ever held role '{target_role}'.")
    
    # 3. Validation
    if len(users_with_role) == 0:
        print(f"Warning: No users found with role '{target_role}'. Check case sensitivity or spelling.")
        return

    non_insiders = users_with_role - insider_users
    
    print("-" * 50)
    print(f"Users with role '{target_role}': {len(users_with_role)}")
    print(f"Insiders in that group: {len(users_with_role & insider_users)}")
    print(f"Non-insiders in that group: {len(non_insiders)}")
    print("-" * 50)
    
    if len(non_insiders) == 0:
        print(f"CONFIRMED: Role '{target_role}' is EXCLUSIVE to Insider Users.")
    else:
        print(f"DISPROVED: Role '{target_role}' is held by {len(non_insiders)} normal users.")
        # print(f"Sample normal users: {list(non_insiders)[:5]}")

if __name__ == "__main__":
    main()
