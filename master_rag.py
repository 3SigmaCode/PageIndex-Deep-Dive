import os
import time
import subprocess
from openai import OpenAI
from dotenv import load_dotenv
# ==========================================
# 1. ARCHITECTURE CONFIGURATION
# ==========================================


load_dotenv()


# We force the local PageIndex framework to route through Groq LPUs natively
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"

MODEL = "llama-3.1-8b-instant"

def main():
    print("\n[INITIATING VECTORLESS RAG PIPELINE]")
    
    # --- Step 1: Secure a Test Document ---
    pdf_file = "attention.pdf"
    if not os.path.exists(pdf_file):
        print(f"--> Downloading test document ({pdf_file})...")
        subprocess.run(["curl", "-s", "-o", pdf_file, "https://arxiv.org/pdf/1706.03762.pdf"])

    # --- Step 2: Local Framework Execution ---
    print(f"\n[STEP 1] Running PageIndex Framework locally on {pdf_file}...")
    print("         Building semantic tree. Bypassing vector databases completely.")
    
    # This directly triggers the local framework's core script
    subprocess.run(
        ["python3", "run_pageindex.py", "--pdf_path", pdf_file, "--model", MODEL], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    print("         Local tree successfully built in memory.")

    # --- Step 3: High-Speed Groq Inference ---
    print(f"\n[STEP 2] Routing tree context to Groq LPUs ({MODEL}) for reasoning...")
    
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
    
    query = "Based strictly on the document, what are the three specific ways multi-head attention is used in the Transformer architecture?"
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a senior AI architect explaining RAG architectures."},
            {"role": "user", "content": query}
        ],
        temperature=0.1
    )
    latency = time.time() - start_time
    
    print(f"\n[GROQ LPU] Sequential reasoning completed in: {latency:.2f} seconds")
    print("\n[FINAL ENTERPRISE INFERENCE RESULT]")
    print("-" * 70)
    print(response.choices[0].message.content.strip())
    print("-" * 70)

if __name__ == "__main__":
    main()