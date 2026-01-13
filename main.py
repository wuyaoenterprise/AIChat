import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from supabase import create_client, Client
from PIL import Image
import io
import base64

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="双核心 AI 聚合站 Pro", page_icon="📸", layout="wide")

st.title("🤖 双核心 AI 聚合终端 Pro")
st.markdown("### ChatGPT (OpenAI) | Gemini (Google) | 📸 视觉分析版")

# ==========================================
# 2. 安全与连接
# ==========================================
try:
    OPENAI_KEY = st.secrets["keys"]["openai_api_key"]
    GOOGLE_KEY = st.secrets["keys"]["google_api_key"]
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except Exception as e:
    st.error("❌ 缺少配置！请检查 Secrets。")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = init_supabase()

# ==========================================
# 3. 真正的谷歌登录逻辑
# ==========================================
user_email = None

try:
    # 只要 App 设为 Private，这里就能自动拿到真实邮箱
    if st.user.email:
        user_email = st.user.email
    elif st.experimental_user.email:
        user_email = st.experimental_user.email
except:
    pass

if user_email:
    st.sidebar.success(f"👤 已登录: {user_email}")
else:
    # 如果没开 Private 或者在本地，显示提示
    st.warning("⚠️ 检测到当前为【访客/测试模式】")
    st.info("💡 要启用真正的谷歌登录，请在 Streamlit Cloud 设置中将 App 设为 'Private'。")
    user_email = st.sidebar.text_input("测试邮箱 (本地调试用):", "test@example.com")

if not user_email:
    st.stop()

# ==========================================
# 4. 历史记录 (只存文本)
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
        # 图片数据太大，不存入数据库，只存文本提示
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
# 5. 侧边栏与图片上传
# ==========================================
with st.sidebar:
    st.markdown("---")
    model_choice = st.radio("🧠 选择大脑:", ("ChatGPT-5", "Gemini 3 Pro"), index=1)
    
    st.markdown("### 📸 图片分析")
    uploaded_file = st.file_uploader("上传图片 (支持 JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    user_image = None
    if uploaded_file:
        # 将上传的文件转换为 PIL 图片对象
        user_image = Image.open(uploaded_file)
        st.image(user_image, caption="已上传", use_container_width=True)

    st.markdown("---")
    if "messages" not in st.session_state or st.button("🔄 刷新记录"):
        st.session_state["messages"] = load_history(user_email)
    
    if st.button("🗑️ 清空记录"):
        clear_history(user_email)

# ==========================================
# 6. AI 核心逻辑 (带图片处理)
# ==========================================

def get_gemini_response(messages, image=None):
    genai.configure(api_key=GOOGLE_KEY)
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    # 构造历史
    gemini_history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
    
    chat = model.start_chat(history=gemini_history)
    
    try:
        if image:
            # 如果有图，发送 [文本, 图片]
            response = chat.send_message([messages[-1]["content"], image], stream=True)
        else:
            response = chat.send_message(messages[-1]["content"], stream=True)
        return response
    except Exception as e:
        return f"Gemini Error: {str(e)}"

def get_chatgpt_response(messages, image=None):
    client = OpenAI(api_key=OPENAI_KEY)
    
    # 准备发送的消息列表
    api_messages = list(messages)
    
    # 如果有图片，需要对最新的一条消息进行改造 (转 Base64)
    if image:
        # 1. 图片转 Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # 2. 替换最后一条消息为“多模态”格式
        last_content = api_messages[-1]["content"]
        api_messages[-1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": last_content},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
            ]
        }

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=api_messages,
            stream=True
        )
        return response
    except Exception as e:
        return f"ChatGPT Error: {str(e)}"

# ==========================================
# 7. 聊天交互区
# ==========================================

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入问题... (如有图片请先在左侧上传)"):
    
    # 1. 组合显示内容
    display_content = prompt
    if user_image:
        display_content = f"[图片上传] {prompt}"
        
    # 2. 显示用户消息
    with st.chat_message("user"):
        st.markdown(display_content)
        if user_image:
            st.image(user_image, width=200)
            
    st.session_state["messages"].append({"role": "user", "content": display_content})
    # 存入数据库
    save_message(user_email, model_choice, "user", display_content)

    # 3. AI 回复
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 调用 AI (传入图片)
        if model_choice == "ChatGPT-5":
            stream = get_chatgpt_response(st.session_state["messages"], user_image)
            # 处理 GPT 流
            if isinstance(stream, str):
                response_placeholder.error(stream)
                full_response = stream
            else:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

        elif model_choice == "Gemini 3 Pro":
            stream = get_gemini_response(st.session_state["messages"], user_image)
            # 处理 Gemini 流
            if isinstance(stream, str):
                response_placeholder.error(stream)
                full_response = stream
            else:
                for chunk in stream:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

    # 4. 保存 AI 回复
    st.session_state["messages"].append({"role": "assistant", "content": full_response})
    save_message(user_email, model_choice, "assistant", full_response)
    
    # 对话结束后，提醒用户如果不需要分析下一张图，记得点×
    if user_image:
        st.toast("✅ 图片已分析。如需分析新图片，请先在左侧移除旧图片。", icon="📸")

