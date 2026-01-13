import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from supabase import create_client, Client
from PIL import Image
import io
import base64
import time
import json
from streamlit_oauth import OAuth2Component

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="双核心 AI 聚合站 Pro", page_icon="📸", layout="wide")

# ==========================================
# 2. 安全与连接 (加载 Secrets)
# ==========================================
try:
    OPENAI_KEY = st.secrets["keys"]["openai_api_key"]
    GOOGLE_KEY = st.secrets["keys"]["google_api_key"]
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    
    # OAuth 配置
    CLIENT_ID = st.secrets["oauth"]["client_id"]
    CLIENT_SECRET = st.secrets["oauth"]["client_secret"]
    REDIRECT_URI = st.secrets["oauth"]["redirect_uri"]
except Exception as e:
    st.error(f"❌ 缺少配置！请检查 .streamlit/secrets.toml。错误信息: {e}")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = init_supabase()

# ==========================================
# 3. 真正的 Google OAuth 2.0 登录逻辑
# ==========================================
st.title("🤖 双核心 AI 聚合终端 Pro")

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

if not st.session_state["user_email"]:
    st.markdown("### 🔐 请先登录以解锁 Pro 功能")
    st.info("使用 Google 账号登录，您的对话历史将安全地存储在云端。")
    
    # 初始化 OAuth 组件
    oauth2 = OAuth2Component(
        CLIENT_ID, 
        CLIENT_SECRET, 
        "https://accounts.google.com/o/oauth2/v2/auth", 
        "https://oauth2.googleapis.com/token", 
        "https://oauth2.googleapis.com/token", 
        REDIRECT_URI
    )
    
    # 显示登录按钮
    result = oauth2.authorize_button(
        name="使用 Google 登录", 
        icon="https://www.google.com.tw/favicon.ico", 
        scope="openid email profile", 
        redirect_uri=REDIRECT_URI,
        use_container_width=True
    )
    
    if result and result.get("token"):
        # 解码 ID Token 获取邮箱
        id_token = result["token"]["id_token"]
        # 简单的 Base64 解码 (实际生产环境建议用 jwt 库校验签名，但这里为了轻量化直接解包)
        payload = id_token.split('.')[1]
        padded = payload + '=' * (4 - len(payload) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(padded))
        
        email = decoded.get("email")
        
        if email:
            st.session_state["user_email"] = email
            st.success(f"登录成功！欢迎, {email}")
            time.sleep(1)
            st.rerun()
    
    st.warning("⚠️ 未登录状态下无法使用 AI 功能及查看历史记录。")
    st.stop() # 🛑 阻止下方代码执行，直到登录成功

# --- 以下代码只有登录后才会执行 ---
user_email = st.session_state["user_email"]

# ==========================================
# 4. 历史记录 (Supabase)
# ==========================================
def load_history(email):
    try:
        response = supabase.table("chat_history")\
            .select("*")\
            .eq("user_email", email)\
            .order("created_at", desc=False)\
            .execute()
        messages = []
        for row in response.data:
            messages.append({"role": row["role"], "content": row["content"]})
        return messages
    except:
        return []

def save_message(email, model, role, content):
    try:
        if content.startswith("[图片上传]"):
            save_content = "[用户上传了一张图片进行分析]"
        else:
            save_content = content
            
        supabase.table("chat_history").insert({
            "user_email": email,
            "model_name": model,
            "role": role,
            "content": save_content
        }).execute()
    except Exception as e:
        print(f"Save error: {e}")

def clear_history(email):
    supabase.table("chat_history").delete().eq("user_email", email).execute()
    st.session_state["messages"] = []
    st.rerun()

# ==========================================
# 5. 侧边栏与控制台
# ==========================================
with st.sidebar:
    st.success(f"👤 已登录: {user_email}")
    if st.button("🚪 退出登录"):
        st.session_state["user_email"] = None
        st.rerun()
        
    st.markdown("---")
    # 严格按照你要求的模型名称
    model_choice = st.radio("🧠 选择大脑:", ("gpt-5", "gemini-3-flash-preview"), index=1)
    
    st.markdown("### 📸 图片分析")
    uploaded_file = st.file_uploader("上传图片 (支持 JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    user_image = None
    if uploaded_file:
        user_image = Image.open(uploaded_file)
        st.image(user_image, caption="已上传", use_container_width=True)

    st.markdown("---")
    if "messages" not in st.session_state or st.button("🔄 刷新记录"):
        st.session_state["messages"] = load_history(user_email)
    
    if st.button("🗑️ 清空记录"):
        clear_history(user_email)

# ==========================================
# 6. AI 核心逻辑 (Gemini & GPT)
# ==========================================

def get_gemini_response(messages, image=None):
    genai.configure(api_key=GOOGLE_KEY)
    # 严格使用你指定的模型名称
    model_name = 'gemini-3-flash-preview'
    
    try:
        model = genai.GenerativeModel(model_name)
    except Exception:
        # 如果该名称报错（因为Google还没发布3.0），为了不让程序崩溃，这里做一个极其隐蔽的fallback，
        # 但既然你强烈要求不要乱改，我保留你的字符串。如果API报错，请检查模型名称是否有效。
        model = genai.GenerativeModel(model_name)

    gemini_history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
    
    chat = model.start_chat(history=gemini_history)
    
    try:
        if image:
            response = chat.send_message([messages[-1]["content"], image], stream=True)
        else:
            response = chat.send_message(messages[-1]["content"], stream=True)
        return response
    except Exception as e:
        return f"Gemini Error ({model_name}): {str(e)}"

def get_chatgpt_response(messages, image=None):
    client = OpenAI(api_key=OPENAI_KEY)
    
    api_messages = list(messages)
    
    if image:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        last_content = api_messages[-1]["content"]
        api_messages[-1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": last_content},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
            ]
        }

    try:
        # 严格使用你指定的模型名称
        response = client.chat.completions.create(
            model="gpt-5", 
            messages=api_messages,
            stream=True
        )
        return response
    except Exception as e:
        return f"ChatGPT Error (gpt-5): {str(e)}"

# ==========================================
# 7. 聊天交互区
# ==========================================
st.markdown(f"#### 当前模型: `{model_choice}`")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入问题... (如有图片请先在左侧上传)"):
    
    display_content = prompt
    if user_image:
        display_content = f"[图片上传] {prompt}"
        
    with st.chat_message("user"):
        st.markdown(display_content)
        if user_image:
            st.image(user_image, width=200)
            
    st.session_state["messages"].append({"role": "user", "content": display_content})
    save_message(user_email, model_choice, "user", display_content)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        if model_choice == "gpt-5":
            stream = get_chatgpt_response(st.session_state["messages"], user_image)
            if isinstance(stream, str):
                response_placeholder.error(stream)
                full_response = stream
            else:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

        elif model_choice == "gemini-3-flash-preview":
            stream = get_gemini_response(st.session_state["messages"], user_image)
            if isinstance(stream, str):
                response_placeholder.error(stream)
                full_response = stream
            else:
                for chunk in stream:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})
    save_message(user_email, model_choice, "assistant", full_response)
    
    if user_image:
        st.toast("✅ 图片已分析。如需分析新图片，请先在左侧移除旧图片。", icon="📸")

