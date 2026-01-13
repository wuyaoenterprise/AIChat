import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from supabase import create_client, Client
from PIL import Image
import io
import base64
import time
import json
# 👇【修正1】这里之前多写了'it'，已修正
from streamlit_oauth import OAuth2Component
import PyPDF2

# ==========================================
# 0. 内置核心提示词 (Persona)
# ==========================================
STOCK_ANALYST_PROMPT = """
# Role: 华尔街资深量化宏观交易员 (Senior Quant-Macro Trader)

## Core Philosophy
你不是一般的金融顾问，你是激进侧重短期Alpha收益的交易员。你的信条是：“市场永远是对的，但大多数人的解读是错的。”你擅长利用多维数据寻找不对称的风险收益比（Asymmetric Risk/Reward）。

## Analysis Framework (必须严格执行的四维分析法)
在分析任何标的（股票、加密货币、期权）时，必须按顺序执行以下深度扫描：

### 1. 🔍 消息面与情绪 (Sentiment & Catalyst)
- **新闻解析**：最近是否有财报、并购、监管变动？要解读“市场预期差”。
- **情绪温度**：当前是贪婪还是恐惧？是否存在“Sell the news”的风险？
- **主力动向**：机构资金（Smart Money）是在吸筹还是派发？

### 2. 📈 技术面解剖 (Technical Deep Dive)
- **趋势结构**：基于道氏理论或艾略特波浪，当前处于上升、下跌还是盘整？
- **关键指标**：
  - **动能**：RSI 是否背离？MACD 柱状图变化？
  - **均线**：价格相对于 MA20, MA50, MA200 的位置？
  - **形态**：是否有头肩底、旗形整理、双顶等经典形态？
- **量价关系**：上涨缩量还是放量？关键位置是否有天量支撑？

### 3. 📜 历史走势与分形 (Historical & Seasonal)
- **历史分形**：当前的走势是否像历史上某个时期的翻版？
- **季节性**：该标的在当前月份/季度的历史表现如何？
- **波动率**：当前的 IV (隐含波动率) 处于历史高位还是低位？

### 4. 💰 估值与基本面 (Fundamental Logic - 短期视角)
- 对于短期交易，只关注催化剂（Catalyst）和估值修复空间。

## Output Rules (输出铁律)
1. **拒绝废话**：严禁输出“投资有风险”等合规性废话。
2. **观点鲜明**：必须给出【看多 Bullish】、【看空 Bearish】或【观望 Neutral】的明确结论。
3. **数字导向**：涉及支撑压力时，必须给出具体价格数字。

## Response Format (最终输出格式)
请严格按照以下Markdown格式输出：
---
### 🎯 [股票代码] 深度交易综述
**交易信号**：🟢 激进做多 / 🔴 坚决做空 / 🟡 观望等待 (置信度: X%)

#### 1. 核心逻辑
> 一句话总结

#### 2. 多维共振分析
* **🕵️ 消息/情绪**：...
* **📊 技术/量价**：...
* **⏳ 历史/趋势**：...

#### 3. 操盘计划
* **入场区间**：$XXX - $XXX
* **第一止盈位**：$XXX
* **止损位**：$XXX
* **盈亏比**：1 : X

#### 4. 风险警示
* 跌破 $XXX 立即离场。
---
#### 5. 个人口语化建议
(用大白话、像朋友一样告诉我你会怎么做)
"""

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="双核心 AI 聚合站 Pro", page_icon="📈", layout="wide")

# ==========================================
# 2. 安全与连接
# ==========================================
try:
    OPENAI_KEY = st.secrets["keys"]["openai_api_key"]
    GOOGLE_KEY = st.secrets["keys"]["google_api_key"]
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    # 👇【修正2】如果不加这段，就会报 Screenshot 2 的错
    CLIENT_ID = st.secrets["oauth"]["client_id"]
    CLIENT_SECRET = st.secrets["oauth"]["client_secret"]
    REDIRECT_URI = st.secrets["oauth"]["redirect_uri"]
except Exception as e:
    st.error(f"❌ 缺少配置！请检查 Secrets。错误详情: {e}")
    if "oauth" in str(e):
        st.info("👉 你忘记在 Secrets 里添加 [oauth] 部分了！")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = init_supabase()

# ==========================================
# 3. Google OAuth
# ==========================================
st.title("🤖 双核心 AI 聚合终端 Pro (交易员版)")

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

if not st.session_state["user_email"]:
    st.markdown("### 🔐 请先登录")
    oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, "https://accounts.google.com/o/oauth2/v2/auth", "https://oauth2.googleapis.com/token", "https://oauth2.googleapis.com/token", REDIRECT_URI)
    result = oauth2.authorize_button(name="使用 Google 登录", icon="https://www.google.com.tw/favicon.ico", scope="openid email profile", redirect_uri=REDIRECT_URI, use_container_width=True)
    
    if result and result.get("token"):
        id_token = result["token"]["id_token"]
        payload = id_token.split('.')[1]
        padded = payload + '=' * (4 - len(payload) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(padded))
        st.session_state["user_email"] = decoded.get("email")
        st.rerun()
    st.stop()

user_email = st.session_state["user_email"]

# ==========================================
# 4. 历史记录
# ==========================================
def load_history(email):
    try:
        response = supabase.table("chat_history").select("*").eq("user_email", email).order("created_at", desc=False).execute()
        return [{"role": r["role"], "content": r["content"]} for r in response.data]
    except: return []

def save_message(email, model, role, content):
    try:
        save_content = content[:2000] + "... [截断]" if len(content) > 2000 else content
        supabase.table("chat_history").insert({"user_email": email, "model_name": model, "role": role, "content": save_content}).execute()
    except Exception as e: print(f"Save error: {e}")

def clear_history(email):
    supabase.table("chat_history").delete().eq("user_email", email).execute()
    st.session_state["messages"] = []
    st.rerun()

# ==========================================
# 5. 侧边栏 (控制中心)
# ==========================================
with st.sidebar:
    st.success(f"👤 {user_email}")
    if st.button("🚪 退出"):
        st.session_state["user_email"] = None
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 🧠 大脑与模式")
    model_choice = st.radio("选择模型:", ("gpt-5", "gemini-3-flash-preview"), index=1)
    
    # 模式切换
    mode_choice = st.selectbox(
        "设定身份:", 
        ["🤖 通用助手", "📈 华尔街量化交易员"]
    )
    
    if mode_choice == "📈 华尔街量化交易员":
        st.caption("✅ 交易员模式已激活")
    
    st.markdown("---")
    st.markdown("### 📂 超级文件上传")
    # 这里 accept_multiple_files=True 允许你按住 Ctrl 选多张
    uploaded_files = st.file_uploader(
        "支持 PDF/图片/CSV/代码", 
        type=["jpg", "png", "jpeg", "pdf", "txt", "csv", "py", "md", "json"],
        accept_multiple_files=True
    )
    
    current_images = []
    current_text_context = ""
    
    if uploaded_files:
        st.caption(f"已加载 {len(uploaded_files)} 个文件")
        for f in uploaded_files:
            try:
                # A. 图片处理
                if f.type.startswith("image"):
                    img = Image.open(f)
                    # 压缩大图，防止 API 报错
                    img.thumbnail((1024, 1024)) 
                    current_images.append(img)
                
                # B. PDF 处理
                elif f.type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(f)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()
                    current_text_context += f"\n\n--- PDF内容: {f.name} ---\n{pdf_text[:10000]}... (PDF过长截取)\n"
                    
                # C. 文本处理
                else:
                    stringio = io.StringIO(f.getvalue().decode("utf-8", errors='ignore'))
                    current_text_context += f"\n\n--- 文件: {f.name} ---\n{stringio.read()}\n"
            except Exception as e:
                st.error(f"文件 {f.name} 解析失败: {e}")

    # 👇【修正3】修复 Screenshot 3 的错误
    # 去掉了 caption 参数，彻底解决"Cannot pair captions"的报错
    if current_images:
        with st.expander(f"已解析 {len(current_images)} 张图片 (点击查看)", expanded=False):
            st.image(current_images[:4], width=150) 
            if len(current_images) > 4:
                st.caption("...及更多图片")

    st.markdown("---")
    if st.button("🗑️ 清空记录"): clear_history(user_email)

# ==========================================
# 6. AI 核心逻辑
# ==========================================
def get_gemini_response(messages, images=None, system_instruction=None):
    genai.configure(api_key=GOOGLE_KEY)
    model = genai.GenerativeModel('gemini-3-flash-preview') 
    
    gemini_history = []
    # 如果有系统指令，注入到对话开头
    if system_instruction:
         gemini_history.append({"role": "user", "parts": [f"System Instruction: {system_instruction}"]})
         gemini_history.append({"role": "model", "parts": ["Understood."]})

    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
    
    chat = model.start_chat(history=gemini_history)
    
    try:
        prompt_content = [messages[-1]["content"]]
        if images: prompt_content.extend(images)
        return chat.send_message(prompt_content, stream=True)
    except Exception as e: return f"Gemini Error: {e}"

def get_chatgpt_response(messages, images=None, system_instruction=None):
    client = OpenAI(api_key=OPENAI_KEY)
    api_messages = list(messages)
    
    if system_instruction:
        api_messages.insert(0, {"role": "system", "content": system_instruction})

    if images:
        last_msg = api_messages[-1]
        content_list = [{"type": "text", "text": last_msg["content"]}]
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}})
        api_messages[-1] = {"role": "user", "content": content_list}

    try:
        return client.chat.completions.create(model="gpt-5", messages=api_messages, stream=True)
    except Exception as e: return f"GPT Error: {e}"

# ==========================================
# 7. 聊天交互
# ==========================================
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入指令 / 股票代码..."):
    
    full_prompt_text = prompt
    display_text = prompt
    
    if current_text_context:
        full_prompt_text += f"\n\n【参考文件内容】:{current_text_context}"
        display_text += " [📄 附带了文件资料]"
    if current_images:
        display_text = f"[🖼️ {len(current_images)} 张图片] {display_text}"

    system_prompt = STOCK_ANALYST_PROMPT if mode_choice == "📈 华尔街量化交易员" else None

    with st.chat_message("user"):
        st.markdown(display_text)
        if current_images: st.image(current_images[:4], width=150) # 这里也去掉了caption
            
    st.session_state["messages"].append({"role": "user", "content": full_prompt_text})
    save_message(user_email, model_choice, "user", display_text)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""
        
        if model_choice == "gpt-5":
            stream = get_chatgpt_response(st.session_state["messages"], current_images, system_prompt)
        else:
            stream = get_gemini_response(st.session_state["messages"], current_images, system_prompt)

        if isinstance(stream, str):
            placeholder.error(stream)
            full_res = stream
        else:
            for chunk in stream:
                content = chunk.choices[0].delta.content if model_choice == "gpt-5" else chunk.text
                if content:
                    full_res += content
                    placeholder.markdown(full_res + "▌")
            placeholder.markdown(full_res)

    st.session_state["messages"].append({"role": "assistant", "content": full_res})
    save_message(user_email, model_choice, "assistant", full_res)
    
    if current_images or current_text_context:
        st.toast("✅ 分析完成，建议移除文件以免干扰下次对话。", icon="💡")
