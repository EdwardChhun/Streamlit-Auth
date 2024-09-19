import streamlit_authenticator as stauth 

hashedPass = hashed_passwords = stauth.Hasher(['abc', 'def']).generate()

print(hashedPass)