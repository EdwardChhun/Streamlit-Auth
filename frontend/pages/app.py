import yaml
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader

# hashed_passwords = stauth.Hasher(['abc', 'def']).generate()

# Depends on where you run the script
with open('../config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
    
# Pre-hashing all plain text passwords once
# Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate( 
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login(captcha=True)

# Checks if authenticated and then access to content
if st.session_state['authentication_status']:
    authenticator.logout() # optional logout button, would have it on menu though
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Some content')
elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')
    
# Creating a reset password widget
if st.session_state['authentication_status']:
    try:
        if authenticator.reset_password(st.session_state['username']):
            st.success('Password modified successfully')
    except Exception as e:
        st.error(e)
        
# Creating a new user
try:
    email, username, name = authenticator.register_user(captcha=False, pre_authorization=False)
    if email:
        st.success('User registered successfully')
         # Add the new user to the config
        config['credentials']['usernames'][username] = {
            'email': email,
            'name': name,
            'password': stauth.Hasher([authenticator._credentials['usernames'][username]['password']]).generate()[0]
        }
        st.success('User registered successfully')
except Exception as e:
    st.error(e)
    
# Forgot user name
try:
    username_of_forgotten_username, email_of_forgotten_username = authenticator.forgot_username()
    if username_of_forgotten_username:
        st.success('Username to be sent securely')
        # The developer should securely transfer the username to the user.
    elif username_of_forgotten_username == False:
        st.error('Email not found')
except Exception as e:
    st.error(e)
    
# Update user details
if st.session_state['authentication_status']:
    try:
        if authenticator.update_user_details(st.session_state['username']):
            st.success('Entries updated successfully')
    except Exception as e:
        st.error(e)

# Update config file
with open('../config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)