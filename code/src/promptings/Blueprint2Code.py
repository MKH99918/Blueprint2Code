from typing import List
import tiktoken
import os
import json
import re
import sys
import time

from copy import deepcopy




class Blueprint2Code(BaseStrategy):
    def __init__(
            self,
            k: int = 3,
            t: int = 5,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.t = t

    def xml_to_dict(self, element):
        result = {}
        for child in element:
            if child:
                child_data = self.xml_to_dict(child)
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = [result[child.tag], child_data]
                else:
                    result[child.tag] = child_data
            else:
                result[child.tag] = child.text
        return result

    def parse_xml(self, response: str) -> dict:
        if '```xml' in response:
            response = response.replace('```xml', '')
        if '```' in response:
            response = response.replace('```', '')

        try:
            root = ET.fromstring(response)
        except:
            try:
                root = ET.fromstring('<root>\n' + response + '\n</root>')
            except:
                root = ET.fromstring('<root>\n' + response)

        result = self.xml_to_dict(root)


        if "problem" in result:
            if not isinstance(result["problem"], list):
                result["problem"] = [result["problem"]]
            for i, problem in enumerate(result["problem"]):
                if isinstance(problem, str):
                    result["problem"][i] = {
                        "description": problem,
                        "code": "",
                        "planning": "",
                        "techniques": ""
                    }

        return result

    def parse_code(self, response: str) -> str:
        if "```" not in response:
            return response

        # 优化代码解析逻辑
        code_pattern = r'```(?:[a-zA-Z0-9#+]*\n)?([\s\S]*?)```'
        code_blocks = re.findall(code_pattern, response, re.DOTALL)

        if code_blocks:
            # 取最后一个代码块
            return code_blocks[-1].strip()
        return response

    @staticmethod
    def trim_text(text: str, trimmed_text: str):
        return text.replace(trimmed_text, '').strip()

    @staticmethod
    def replace_tag(text: str, tag: str):
        if f'<{tag}><![CDATA[' in text and f']]></{tag}>' in text:
            return text
        else:
            return text.replace(f'<{tag}>', f'<{tag}><![CDATA[').replace(f'</{tag}>', f']]></{tag}>').strip()

    @staticmethod
    def get_sample_io_str(sample_io: any) -> str:
        if len(sample_io) > 0:
            if type(sample_io[0]) == str:
                return "\n".join(sample_io)
            if type(sample_io[0]) == dict:
                return "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io])
        return sample_io

    def run_single_pass(self, item: dict):
        print("", flush=True)

        # 1. 增强检索智能体：学习代码生成技巧
        input_kb_exemplars = [
            {
                "role": "user",
                "content": f"""Given a problem, provide relevant problems and learn code generation techniques from them. Also identify the algorithm behind the problem and explain its tutorial.
# Problem:
{self.data.get_prompt(item)}

# Exemplars:
Recall {self.k} relevant and distinct problems (different from the given problem). For each problem:
1. Describe it concisely
2. Generate {self.language} code step by step to solve that problem
3. Analyze and extract 1-3 key code generation techniques used in the solution
4. Generate a detailed planning to solve that problem

# Algorithm:

----------------
Important:
Your response must follow the following xml format:

<root>
<problem>
<description>
# Describe the problem concisely.
</description>
<code>
# Let's think step by step to solve this problem in {self.language} programming language.
</code>
<techniques>
# Extract 1-3 key code generation techniques used in this solution.
</techniques>
<planning>
# Detailed planning to solve this problem.
</planning>
</problem>

# Add more problems here...

<algorithm>
# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, etc.) that needs to be used to solve the original problem.
# Write a useful tutorial about the identified algorithm. Provide a high-level generic tutorial for solving this type of problem. Do not generate code.
</algorithm>

<learned_techniques>
# Summarize the key code generation techniques learned from all the examples (3-5 techniques).
</learned_techniques>
</root>
""",
            },
        ]

        print("\n\n________________________")
        print("Input for knowledge base and exemplars: ")
        print(input_kb_exemplars[0]['content'], flush=True)

        response, pr_tok, com_tok = self.gpt_chat(
            processed_input=input_kb_exemplars
        )
        item['api_calls'] = item.get('api_calls', 0) + 1

        # 后处理
        response = self.replace_tag(response, 'algorithm')
        response = self.replace_tag(response, 'description')
        response = self.replace_tag(response, 'code')
        response = self.replace_tag(response, 'planning')
        response = self.replace_tag(response, 'techniques')
        response = self.replace_tag(response, 'learned_techniques')

        print("\n\n________________________")
        print("Response from knowledge base and exemplars: ")
        print(response, flush=True)

        response = self.parse_xml(response)


        problems = response.get("problem", [])
        if not isinstance(problems, list):
            problems = [problems]

        algorithm_prompt = f"## Relevant Algorithm: {response.get('algorithm', '')}"
        learned_techniques = f"## Learned Code Generation Techniques: {response.get('learned_techniques', '')}"
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"

        plannings = []
        for example_no, example in enumerate(problems, start=1):
            # 确保示例是字典格式
            if isinstance(example, str):
                example = {"description": example, "code": "", "planning": "", "techniques": ""}

            example_problem = example.get("description", "")
            example_planning = example.get("planning", "")
            example_techniques = example.get("techniques", "")

            # 2. 增强计划智能体：生成更详细的计划
            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": f"""Given a competitive programming problem, generate a detailed, step-by-step plan to solve it.
# Example Problem:
{example_problem}

# Example Techniques:
{example_techniques}

# Example Planning:
{example_planning}

# Algorithm:
{algorithm_prompt}

# Learned Techniques:
{learned_techniques}

# Problem to Solve:
{self.data.get_prompt(item)}

# Sample Test Cases:
{sample_io_prompt}

# Detailed Planning:
Create a detailed, step-by-step plan to solve the problem. Structure your plan as:
1. Step 1: [Description of first step]
2. Step 2: [Description of second step]
...
n. Step n: [Description of final step]

Important: 
- Be specific and concrete in each step
- Consider edge cases and input/output handling
- Include time and space complexity considerations
- Do not generate code, only the planning
"""
                }
            ]

            print("\n\n________________________")
            print(f"Input for our problem planning using example: {example_no}: ")
            print(input_for_problem_planning[0]['content'], flush=True)

            planning, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_problem_planning
            )
            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            print("\n\n________________________")
            print("Response from our problem planning: ")
            print(planning, flush=True)

            # 计划验证
            input_for_planning_verification = [
                {
                    "role": "user",
                    "content": f"""Evaluate the following plan for solving the problem. Provide a confidence score (0-100) and explain your reasoning.
# Problem:
{self.data.get_prompt(item)}

# Proposed Plan:
{planning}

# Evaluation Criteria:
1. Completeness: Does the plan cover all aspects of the problem?
2. Correctness: Is the algorithmic approach sound?
3. Feasibility: Can the plan be implemented effectively?
4. Edge Cases: Does the plan consider boundary conditions?
5. Efficiency: Does the plan consider time and space complexity?

# Your Response:
<root>
<analysis>
# Detailed analysis of the plan's strengths and weaknesses
</analysis>
<confidence>
# Confidence score (0-100 integer) based on the above criteria
</confidence>
</root>
"""
                }
            ]

            print("Input for planning verification: ")
            print(input_for_planning_verification[0]['content'], flush=True)

            verification_res, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_planning_verification
            )
            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            verification_res = self.replace_tag(verification_res, 'analysis')
            verification_res = self.replace_tag(verification_res, 'confidence')
            verification_res = self.parse_xml(verification_res)


            confidence_score = 0
            try:
                confidence_text = verification_res.get('confidence', '0')
                confidence_score = int(re.search(r'\d+', confidence_text).group())
                confidence_score = max(0, min(100, confidence_score))
            except:
                confidence_score = 50  # 默认值

            print("Response from planning verification: ")
            print(f"Analysis: {verification_res.get('analysis', '')}")
            print(f"Confidence: {confidence_score}")

            plannings.append((
                planning,
                confidence_score,
                example
            ))

        # 按置信度排序计划
        plannings.sort(key=lambda x: x[1], reverse=True)

        # 标准输入/输出提示
        if type(self.data) in [APPSDataset]:
            std_input_prompt = "## Note: Strictly follow the input and output format. Take input from stdin and output to stdout. If writing a function, after the function definition, take input using `input()`, call the function, and print the result. Avoid extra print statements."
        else:
            std_input_prompt = ""

        # 3. 代码生成智能体（保持原有结构但优化提示词）
        for planning_with_ex in plannings:
            planning, confidence, example = planning_with_ex

            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": f"""Generate {self.language} code to solve the following problem based on the provided plan.
# Problem:
{self.data.get_prompt(item)}

# Planning:
{planning}

# Sample Test Cases:
{sample_io_prompt}

# Learned Techniques:
{learned_techniques}

# Algorithm:
{algorithm_prompt}

# Instructions:
1. Implement the solution exactly as per the planning
2. Add comments to explain key steps
3. Handle edge cases appropriately
4. {std_input_prompt}

# Your Response:
Generate only the {self.language} code. Do not include any explanations.
"""
                }
            ]

            print("\n\n________________________")
            print("Input for final code generation: ")
            print(input_for_final_code_generation[0]['content'], flush=True)

            code, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_final_code_generation
            )
            item['api_calls'] += 1
            code = self.parse_code(code)
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            print("\n\n________________________")
            print("Response from final code generation: ")
            print(code, flush=True)

            # 4. 增强调试智能体：提供更详细的错误分析
            response_record = f"## Planning: {planning}\n## Code:\n```\n{code}\n```"
            passed = False

            for i in range(1, self.t + 1):
                passed, test_log = self.data.evaluate_sample_io(
                    item,
                    code,
                    self.language
                )

                if passed:
                    break

                print(f"Input for improving code generation: {i}")
                input_for_improving_code = [
                    {
                        "role": "user",
                        "content": f"Given a competitive programming problem you have generated {self.language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{response}\n## Test Report:\n{test_log}\n## Modified Planning:\n## Let's think step by step to modify {self.language} Code for solving this problem.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain the modified planning and then the {self.language} code inside ``` block to solve this problem."
                    }
                ]

                print("\n\n________________________")
                print("Input for improving code generation: ")
                print(input_for_improving_code[0]['content'], flush=True)

                response, pr_tok_1, com_tok_1 = self.gpt_chat(
                    input_for_improving_code
                )
                item['api_calls'] += 1
                # time.sleep(1)

                code = self.parse_code(response)
                pr_tok += pr_tok_1
                com_tok += com_tok_1

                print("\n\n________________________")
                print("Response from improving code generation: ")
                print(response, flush=True)

            # got a code that passed all sample test cases
            if passed:
                break

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok
