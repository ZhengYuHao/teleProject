# test_qwen_api.py
import requests
import json
import time

class QwenAPITester:
    """
    用于测试千问大模型API链接有效性的模块
    """
    
    def __init__(self, base_url):
        """
        初始化测试器
        
        Args:
            base_url (str): 千问大模型的API基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def test_connection(self):
        """
        测试基础连接是否可用
        
        Returns:
            dict: 包含测试结果的字典
        """
        result = {
            'success': False,
            'status_code': None,
            'response_time': None,
            'error': None,
            'details': None
        }
        
        try:
            start_time = time.time()
            response = self.session.get(self.base_url, timeout=10)
            end_time = time.time()
            
            result['status_code'] = response.status_code
            result['response_time'] = round((end_time - start_time) * 1000, 2)  # 毫秒
            
            if response.status_code == 200:
                result['success'] = True
                result['details'] = '连接成功'
            else:
                result['error'] = f'HTTP状态码: {response.status_code}'
                try:
                    result['details'] = response.json()
                except:
                    result['details'] = response.text[:200]  # 只取前200字符
                    
        except requests.exceptions.Timeout:
            result['error'] = '连接超时'
        except requests.exceptions.ConnectionError:
            result['error'] = '连接错误'
        except Exception as e:
            result['error'] = f'未知错误: {str(e)}'
            
        return result
    
    def test_api_endpoints(self):
        """
        测试常见的API端点
        
        Returns:
            dict: 包含各端点测试结果的字典
        """
        endpoints = [
            '/',           # 根路径
            '/health',     # 健康检查
            '/v1/models',  # 模型列表
            '/docs',       # API文档
            '/openapi.json' # OpenAPI规范
        ]
        
        results = {}
        
        for endpoint in endpoints:
            url = self.base_url + endpoint
            result = {
                'success': False,
                'status_code': None,
                'response_time': None,
                'error': None
            }
            
            try:
                start_time = time.time()
                response = self.session.get(url, timeout=5)
                end_time = time.time()
                
                result['status_code'] = response.status_code
                result['response_time'] = round((end_time - start_time) * 1000, 2)
                
                if response.status_code == 200:
                    result['success'] = True
                else:
                    result['error'] = f'HTTP {response.status_code}'
                    
            except Exception as e:
                result['error'] = str(e)
                
            results[endpoint] = result
            
        return results
    
    def test_model_inference(self, model_name=None):
        """
        测试模型推理功能（如果API支持）
        
        Args:
            model_name (str, optional): 模型名称
            
        Returns:
            dict: 推理测试结果
        """
        # 尝试获取模型列表
        try:
            models_response = self.session.get(f"{self.base_url}/v1/models", timeout=5)
            if models_response.status_code == 200:
                models_data = models_response.json()
                if 'data' in models_data and len(models_data['data']) > 0:
                    if not model_name:
                        model_name = models_data['data'][0]['id']
                else:
                    # 如果无法获取模型列表，使用默认名称
                    model_name = model_name or "qwen"
            else:
                model_name = model_name or "qwen"
        except:
            model_name = model_name or "qwen"
        
        # 构造测试请求
        test_payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "你好"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        result = {
            'success': False,
            'status_code': None,
            'response_time': None,
            'error': None,
            'response': None
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            end_time = time.time()
            
            result['status_code'] = response.status_code
            result['response_time'] = round((end_time - start_time) * 1000, 2)
            
            if response.status_code in [200, 201]:
                result['success'] = True
                try:
                    result['response'] = response.json()
                except:
                    result['response'] = response.text[:500]
            else:
                result['error'] = f'HTTP {response.status_code}'
                try:
                    error_detail = response.json()
                    result['error'] += f": {error_detail}"
                except:
                    result['error'] += f": {response.text[:200]}"
                    
        except Exception as e:
            result['error'] = str(e)
            
        return result

def main():
    """
    主函数，用于测试指定的千问API链接
    """
    # 测试的API链接
    api_url = "http://106.227.68.83:8000"
    
    print(f"开始测试千问大模型API: {api_url}")
    print("=" * 50)
    
    # 创建测试器实例
    tester = QwenAPITester(api_url)
    
    # 1. 基础连接测试
    print("1. 基础连接测试...")
    connection_result = tester.test_connection()
    print(f"   结果: {'成功' if connection_result['success'] else '失败'}")
    if connection_result['success']:
        print(f"   响应时间: {connection_result['response_time']} ms")
        print(f"   状态码: {connection_result['status_code']}")
    else:
        print(f"   错误: {connection_result['error']}")
    
    # 2. API端点测试
    print("\n2. API端点测试...")
    endpoints_result = tester.test_api_endpoints()
    for endpoint, result in endpoints_result.items():
        status = "✓" if result['success'] else "✗"
        time_info = f" ({result['response_time']} ms)" if result['response_time'] else ""
        print(f"   {status} {endpoint}: {result['status_code']}{time_info}")
        if not result['success'] and result['error']:
            print(f"      错误: {result['error']}")
    
    # 3. 模型推理测试
    print("\n3. 模型推理测试...")
    inference_result = tester.test_model_inference()
    print(f"   结果: {'成功' if inference_result['success'] else '失败'}")
    if inference_result['success']:
        print(f"   响应时间: {inference_result['response_time']} ms")
        print(f"   状态码: {inference_result['status_code']}")
        if inference_result['response']:
            if isinstance(inference_result['response'], dict):
                # 如果是JSON响应，尝试提取有用信息
                if 'choices' in inference_result['response']:
                    try:
                        content = inference_result['response']['choices'][0]['message']['content']
                        print(f"   响应示例: {content[:100]}{'...' if len(content) > 100 else ''}")
                    except:
                        print(f"   响应: {json.dumps(inference_result['response'], ensure_ascii=False)[:100]}...")
                else:
                    print(f"   响应: {json.dumps(inference_result['response'], ensure_ascii=False)[:100]}...")
            else:
                print(f"   响应: {str(inference_result['response'])[:100]}...")
    else:
        print(f"   错误: {inference_result['error']}")
    
    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    main()