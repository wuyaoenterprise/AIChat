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
st.set_page_config(page_title="双核心 AI 聚合站 Pro", page_icon="📂", layout="wide")

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
    st.error(f"❌ 缺少配置！请检查 Secrets 设置。错误详情: {e}")
    if "oauth" in str(e):
        st.info("💡 提示：看起来你忘记在 Secrets 里添加 [oauth] 部分了。")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = init_supabase()

# ==========================================
# 3. Google OAuth 登录逻辑
# ==========================================
st.title("🤖 双核心 AI 聚合终端 Pro (多文件版)")

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

if not st.session_state["user_email"]:
    st.markdown("### 🔐 请先登录")
    st.info("使用 Google 账号登录以解锁 AI 功能及历史记录。")
    
    oauth2 = OAuth2Component(
        CLIENT_ID, 
        CLIENT_SECRET, 
        "https://accounts.google.com/o/oauth2/v2/auth", 
        "https://oauth2.googleapis.com/token", 
        "https://oauth2.googleapis.com/token", 
        REDIRECT_URI
    )
    
    result = oauth2.authorize_button(
        name="使用 Google 登录", 
        icon="https://www.google.com.tw/favicon.ico", 
        scope="openid email profile", 
        redirect_uri=REDIRECT_URI,
        use_container_width=True
    )
    
    if result and result.get("token"):
        id_token = result["token"]["id_token"]
        payload = id_token.split('.')[1]
        padded = payload + '=' * (4 - len(payload) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(padded))
        
        email = decoded.get("email")
        if email:
            st.session_state["user_email"] = email
            st.success(f"登录成功！欢迎, {email}")
            time.sleep(1)
            st.rerun()
            
    st.warning("⚠️ 请登录后使用。")
    st.stop()

user_email = st.session_state["user_email"]

# ==========================================
# 4. 历史记录
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
        # 简化存储，不存过长的文件内容日志
        if len(content) > 2000:
            save_content = content[:200] + "... [内容过长截断]"
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
# 5. 侧边栏 (文件处理中心)
# ==========================================
with st.sidebar:
    st.success(f"👤 {user_email}")
    if st.button("🚪 退出"):
        st.session_state["user_email"] = None
        st.rerun()
        
    st.markdown("---")
    model_choice = st.radio("🧠 模型:", ("gpt-5", "gemini-3-flash-preview"), index=1)
    
    st.markdown("### 📂 文件上传区")
    # 🔥 核心修改：accept_multiple_files=True，且支持更多格式
    uploaded_files = st.file_uploader(
        "支持图片/文本/代码 (按住Ctrl多选)", 
        type=["jpg", "png", "jpeg", "txt", "csv", "py", "md", "json"],
        accept_multiple_files=True
    )
    
    # 处理文件列表
    current_images = []
    current_text_context = ""
    
    if uploaded_files:
        st.caption(f"已加载 {len(uploaded_files)} 个文件")
        for f in uploaded_files:
            # 1. 如果是图片
            if f.type.startswith("image"):
                img = Image.open(f)
                current_images.append(img)
                with st.expander(f"🖼️ {f.name}", expanded=False):
                    st.image(img, use_container_width=True)
            
            # 2. 如果是文本类文件 (txt, csv, code...)
            else:
                stringio = io.StringIO(f.getvalue().decode("utf-8"))
                file_content = stringio.read()
                # 拼接文件名和内容
                current_text_context += f"\n\n--- 文件名: {f.name} ---\n{file_content}\n"
                with st.expander(f"📄 {f.name}", expanded=False):
                    st.text(file_content[:100] + "...") # 只显示前100字预览

    st.markdown("---")
    if "messages" not in st.session_state or st.button("🔄 刷新"):
        st.session_state["messages"] = load_history(user_email)
    
    if st.button("🗑️ 清空"):
        clear_history(user_email)

# ==========================================
# 6. AI 响应逻辑 (支持多图 + 文本注入)
# ==========================================

def get_gemini_response(messages, images=None):
    """Gemini 支持原生的 List[Image]"""
    genai.configure(api_key=GOOGLE_KEY)
    model = genai.GenerativeModel('gemini-3-flash-preview') 
    
    gemini_history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
    
    chat = model.start_chat(history=gemini_history)
    
    try:
        # 构造发送内容：[文本提示, 图1, 图2, 图3...]
        prompt_content = [messages[-1]["content"]]
        if images:
            prompt_content.extend(images) # 将图片列表追加进去
            
        return chat.send_message(prompt_content, stream=True)
    except Exception as e:
        return f"Gemini Error: {e}"

def get_chatgpt_response(messages, images=None):
    """GPT 需要构造成 content 数组"""
    client = OpenAI(api_key=OPENAI_KEY)
    api_messages = list(messages)
    
    last_msg = api_messages[-1]
    
    # 如果有图片，必须把最后一条消息改成 "多模态" 格式
    if images:
        content_list = [{"type": "text", "text": last_msg["content"]}]
        
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            # 追加每一张图
            content_list.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{img_str}"}
            })
            
        api_messages[-1] = {
            "role": "user",
            "content": content_list
        }

    try:
        return client.chat.completions.create(model="gpt-5", messages=api_messages, stream=True)
    except Exception as e:
        return f"GPT Error: {e}"

# ==========================================
# 7. 聊天界面
# ==========================================
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入问题... (可同时分析多文件)"):
    
    # 1. 组合最终发送给 AI 的文本 (问题 + 文件内容)
    full_prompt_text = prompt
    if current_text_context:
        full_prompt_text += f"\n\n【附带文件内容】:{current_text_context}"
    
    # 2. 组合显示的文本 (用户看到的)
    display_text = prompt
    if current_images:
        display_text = f"[已上传 {len(current_images)} 张图片] {display_text}"
    if current_text_context:
        display_text += " [附带了文本文件]"
        
    # 3. 显示用户消息
    with st.chat_message("user"):
        st.markdown(display_text)
        # 在聊天框里平铺展示上传的缩略图
        if current_images:
            cols = st.columns(len(current_images))
            for idx, img in enumerate(current_images):
                with cols[idx]:
                    st.image(img, use_container_width=True)
    
    # 4. 保存进历史
    st.session_state["messages"].append({"role": "user", "content": full_prompt_text})
    save_message(user_email, model_choice, "user", display_text) # 存数据库时存精简版

    # 5. AI 回复
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""
        
        if model_choice == "gpt-5":
            stream = get_chatgpt_response(st.session_state["messages"], current_images)
            if isinstance(stream, str):
                placeholder.error(stream)
                full_res = stream
            else:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_res += chunk.choices[0].delta.content
                        placeholder.markdown(full_res + "▌")
                placeholder.markdown(full_res)
                
        else: # Gemini
            stream = get_gemini_response(st.session_state["messages"], current_images)
            if isinstance(stream, str):
                placeholder.error(stream)
                full_res = stream
            else:
                for chunk in stream:
                    full_res += chunk.text
                    placeholder.markdown(full_res + "▌")
                placeholder.markdown(full_res)

    st.session_state["messages"].append({"role": "assistant", "content": full_res})
    save_message(user_email, model_choice, "assistant", full_res)
    
    # 提醒用户清理
    if current_images or current_text_context:
        st.toast("✅ 文件分析完毕。如需分析新文件，请在左侧移除旧文件。", icon="📂")
