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
import PyPDF2
import yfinance as yf
# 👇【新增】引入免费搜索库
from duckduckgo_search import DDGS

# ==========================================
# 0. 内置核心提示词 (Persona)
# ==========================================
STOCK_ANALYST_PROMPT = """
# Role: 华尔街资深量化宏观交易员 (Senior Quant-Macro Trader)

## Core Philosophy
你不是一般的机器人，你是拥有20年经验的顶级操盘手。你的风格是**深度、详尽、逻辑缜密**。你不仅给出结论，更看重**逻辑推演的过程**。你拒绝短得像推特一样的回答，你喜欢像写“投资备忘录”一样，把事情的前因后果、市场博弈、宏观背景全部讲清楚。

## Analysis Framework (深度扫描)
在分析时，请务必覆盖以下维度，并尽可能详细地展开：

### 1. 🕵️ 宏观与消息面 (The Narrative)
- **不要只读新闻标题**：结合宏观经济（美联储政策、通胀、地缘政治）来解读个股新闻。
- **博弈分析**：市场现在的预期是什么？这个消息是否已经被Price-in（计价）了？是否存在预期差？
- **机构动向**：Smart Money 在做什么？期权链上的大单在赌什么方向？

### 2. 📈 技术面深度解剖 (Technical Deep Dive)
- **结构与趋势**：从周线看大趋势，从日线看波段。是多头排列还是空头陷阱？
- **量价行为 (Price Action)**：关键位置的成交量如何？有没有原本的支撑变成了压力？
- **指标共振**：RSI、MACD、布林带是否在同一时间指出了同一方向？

### 3. 📜 历史分形与统计 (Historical Context)
- 这只股票在财报季通常怎么走？
- 当前的走势是否像历史上某一次崩盘或暴涨的前夜？

## Output Style (输出风格要求)
1. **像真人一样交谈**：可以使用专业的行话（Alpha, Gamma Squeeze, IV Crush），但要像个导师一样把逻辑讲透。
2. **拒绝简短**：**越详细越好**。不要只列点，要写段落。把每一个分析点的“为什么”讲清楚。
3. **包含具体数据**：提到支撑位、压力位时，必须给出具体价格。

## Response Structure (建议回复结构)
虽然你可以自由发挥，但请确保包含：
- **🎯 核心交易观点** (一针见血的结论)
- **🧐 深度逻辑推演** (这里要长篇大论，把多空逻辑都分析透)
- **📊 关键点位与计划** (具体的入场、止损、止盈数字)
- **💡 像朋友一样的建议** (如果这是你自己的钱，你会怎么操作？)
"""

# ==========================================
# 0.5 工具函数：抓取股票数据
# ==========================================
def get_stock_info(symbol):
    try:
        # 移除可能的多余空格
        symbol = symbol.strip().upper()
        ticker = yf.Ticker(symbol)
        
        # 1. 获取盘中实时/收盘数据 (最近1天, 5分钟级)
        history = ticker.history(period="1d", interval="5m")
        
        # 2. 获取基本信息 (可能包含市盈率、市值等)
        info = ticker.info
        
        if not history.empty:
            latest = history.iloc[-1]
            # 格式化数据字符串
            price_data = f"""
            【{symbol} 实时交易数据快照】
            - 当前价格: {latest['Close']:.2f}
            - 今日开盘: {latest['Open']:.2f}
            - 今日最高: {latest['High']:.2f}
            - 今日最低: {latest['Low']:.2f}
            - 成交量: {latest['Volume']}
            - 市值: {info.get('marketCap', 'N/A')}
            - 盘中走势(最近5个5分钟K线):
            {history.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']].to_string()}
            """
        else:
            price_data = f"【{symbol}】未获取到盘中K线数据 (可能是休市或代码错误)。"

        # 3. 获取最新新闻
        news = ticker.news
        news_str = "\n\n【最新关联新闻】:\n"
        if news:
            for n in news[:3]: # 只取最新的3条
                pub_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(n.get('providerPublishTime', 0)))
                news_str += f"- [{pub_time}] {n.get('title')} (来源: {n.get('publisher')})\n"
        else:
            news_str += "暂无最新即时新闻。"
            
        return price_data + news_str

    except Exception as e:
        return f"尝试抓取 {symbol} 数据时发生错误: {str(e)}"

# ==========================================
# 0.6 工具函数：通用网页搜索 (给 GPT 用)
# ==========================================
def get_web_search_results(query):
    """使用 DuckDuckGo 搜索实时信息"""
    try:
        # 限制搜索结果为 5 条，保证速度
        results = DDGS().text(query, max_results=5)
        if not results:
            return "【搜索结果】未找到相关实时信息。"
        
        search_context = "【🔍 实时互联网搜索结果 (供参考)】:\n"
        for i, res in enumerate(results):
            search_context += f"{i+1}. 标题: {res['title']}\n   摘要: {res['body']}\n   链接: {res['href']}\n\n"
        return search_context
    except Exception as e:
        return f"【搜索错误】无法连接互联网: {str(e)}"

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
# 4.5 初始化消息列表
# ==========================================
if "messages" not in st.session_state:
    if st.session_state.get("user_email"):
        st.session_state["messages"] = load_history(st.session_state["user_email"])
    else:
        st.session_state["messages"] = []
      
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
    model_choice = st.radio("选择模型:", ("gpt-5", "gemini-2.5-pro"), index=1)
    
    # 模式切换
    mode_choice = st.selectbox(
        "设定身份:", 
        ["🤖 通用助手", "📈 华尔街量化交易员"]
    )
    
    if mode_choice == "📈 华尔街量化交易员":
        st.caption("✅ 交易员模式已激活")
        
    # 👇【新增】联网开关
    enable_web = st.toggle("🌍 开启实时联网 (Web Search)", value=True)
    
    st.markdown("---")
    # 侧边栏手动抓取工具
    st.markdown("### 📡 快速行情抓取")
    manual_ticker = st.text_input("输入代码 (如 TSLA):", key="sidebar_ticker").upper()
    if manual_ticker and st.button("🔍 抓取数据并分析"):
        st.session_state["auto_prompt"] = manual_ticker
    
    st.markdown("---")
    st.markdown("### 📂 超级文件上传")
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
    
    # 👇【核心修改】开启官方 Google Search Grounding
    # 使用 gemini-3-flash-preview 以确保兼容性和稳定性
    try:
        model = genai.GenerativeModel('gemini-2.5-pro, tools='google_search_retrieval') 
    except:
        # 降级处理：如果账号不支持搜索，回退到普通模式
        model = genai.GenerativeModel('gemini-2.5-pro')

    gemini_history = []
    if system_instruction:
         gemini_history.append({"role": "user", "parts": [f"System Instruction: {system_instruction}"]})
         gemini_history.append({"role": "model", "parts": ["Understood. I will provide detailed, expert analysis using latest data."]})

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

    # 处理图片
    if images:
        last_msg = api_messages[-1]
        content_list = [{"type": "text", "text": last_msg["content"]}]
        
        for img in images:
            # ✅ 修复 PNG 透明背景报错
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
                
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

# 检查是否有来自侧边栏的自动输入
if "auto_prompt" in st.session_state and st.session_state["auto_prompt"]:
    user_input = st.session_state["auto_prompt"]
    del st.session_state["auto_prompt"]
    prompt = user_input
else:
    prompt = st.chat_input("输入指令 / 股票代码 (如 NVDA)...")

if prompt:
    full_prompt_text = prompt
    display_text = prompt
    
    # 智能识别股票代码
    potential_ticker = prompt.strip().upper()
    is_ticker = (len(potential_ticker) <= 6 and potential_ticker.isalpha()) or ("." in potential_ticker and len(potential_ticker) <= 10)
    
    if is_ticker:
        with st.status(f"📡 正在抓取 {potential_ticker} 实时行情...", expanded=True) as status:
            stock_data = get_stock_info(potential_ticker)
            full_prompt_text += f"\n\n【系统自动抓取的实时行情】:\n{stock_data}"
            display_text += f" [📡 已自动挂载 {potential_ticker} 实时数据]"
            status.update(label="✅ 数据抓取完成", state="complete", expanded=False)
            
    # 👇【新增】如果是普通对话 + 开启联网 + 且不是纯股票查询（股票查询用yfinance更准）
    # 主要针对 GPT 模型，因为 Gemini 已经内置联网
    elif enable_web and model_choice == "gpt-5":
        with st.status(f"🌍 正在搜索全网资料: {prompt[:10]}...", expanded=True) as status:
            web_data = get_web_search_results(prompt)
            full_prompt_text += f"\n\n{web_data}"
            status.update(label="✅ 搜索完成", state="complete", expanded=False)

    # 拼接文件上下文
    if current_text_context:
        full_prompt_text += f"\n\n【参考文件内容】:{current_text_context}"
        display_text += " [📄 附带了文件资料]"
    if current_images:
        display_text = f"[🖼️ {len(current_images)} 张图片] {display_text}"

    system_prompt = STOCK_ANALYST_PROMPT if mode_choice == "📈 华尔街量化交易员" else None

    # 1. 显示用户消息
    with st.chat_message("user"):
        st.markdown(display_text)
        if current_images: 
            st.image(current_images[:4], width=150)
            
    # 2. 保存用户消息
    st.session_state["messages"].append({"role": "user", "content": full_prompt_text})
    save_message(user_email, model_choice, "user", display_text)

    # 3. 生成 AI 回复
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""
        
        # 调用 AI
        if model_choice == "gpt-5":
            stream = get_chatgpt_response(
                st.session_state["messages"], 
                images=current_images, 
                system_instruction=system_prompt
            )
        else:
            stream = get_gemini_response(
                st.session_state["messages"], 
                images=current_images, 
                system_instruction=system_prompt
            )

        # 4. 流式输出处理
        if isinstance(stream, str):
            placeholder.error(stream)
            full_res = stream
        else:
            try:
                for chunk in stream:
                    if model_choice == "gpt-5":
                        content = chunk.choices[0].delta.content
                    else:
                        try:
                            content = chunk.text
                        except ValueError:
                            content = " [⚠️ 安全拦截] "
                    
                    if content:
                        full_res += content
                        placeholder.markdown(full_res + "▌")
            except Exception as e:
                placeholder.error(f"❌ 传输中断: {e}")

        # 5. 最终显示
        if not full_res:
            placeholder.warning("⚠️ AI 无响应，请减少图片或检查网络。")
        else:
            placeholder.markdown(full_res)

        st.session_state["messages"].append({"role": "assistant", "content": full_res})
        save_message(user_email, model_choice, "assistant", full_res)
        
    # 6. 提示
    if current_images or current_text_context:
        st.toast("✅ 分析完成，建议移除文件以免干扰下次对话。", icon="💡")

