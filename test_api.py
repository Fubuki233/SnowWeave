"""
简单的 Gemini API 测试脚本
用于验证 API 密钥是否有效

运行方法:
    python test_api.py
    
或指定 API 密钥:
    python test_api.py YOUR_API_KEY
"""

import sys
from google import genai

def test_api(api_key):
    """测试 API 密钥是否有效"""
    print("="*60)
    print("  Gemini API 测试工具")
    print("="*60)
    print(f"\n正在测试 API 密钥: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # 创建客户端
        print("\n1. 创建客户端...")
        client = genai.Client(api_key=api_key)
        print("   ✅ 客户端创建成功")
        
        # 测试简单的文本生成
        print("\n2. 测试文本生成...")
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents='Say "Hello, API test successful!" in Chinese.'
        )
        print(f"   ✅ 响应: {response.text}")
        
        # 列出可用模型
        print("\n3. 列出可用模型...")
        models = client.models.list()
        video_models = [m for m in models if 'veo' in m.name.lower()]
        
        if video_models:
            print(f"   ✅ 找到 {len(video_models)} 个视频生成模型:")
            for model in video_models:
                print(f"      - {model.name}")
        else:
            print("   ⚠️  未找到 Veo 视频生成模型")
        
        print("\n" + "="*60)
        print("  ✅ API 测试通过！")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("\n可能的原因:")
        print("  1. API 密钥无效或已过期")
        print("  2. 网络连接问题")
        print("  3. API 配额已用完")
        print("\n请检查:")
        print("  - https://aistudio.google.com/apikey")
        print("="*60)
        return False

def main():
    if len(sys.argv) > 1:
        # 从命令行参数获取
        api_key = sys.argv[1]
    else:
        # 从环境变量获取
        import os
        api_key = os.environ.get("GEMINI_API_KEY", "")
        
        if not api_key:
            print("错误: 未提供 API 密钥\n")
            print("用法:")
            print("  python test_api.py YOUR_API_KEY")
            print("\n或设置环境变量:")
            print("  set GEMINI_API_KEY=YOUR_API_KEY  (Windows)")
            print("  export GEMINI_API_KEY=YOUR_API_KEY  (Linux/Mac)")
            sys.exit(1)
    
    success = test_api(api_key)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
