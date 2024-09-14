import os
import gradio as gr
from transformers import pipeline

# Set environment variable to avoid the OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load a model from Hugging Face's Model Hub
generator = pipeline(model="gpt2")

# PlanAgent: Responsible for splitting the task and refining it
class PlanAgent:
    def __init__(self):
        self.tasks = []
    
    def break_into_subtasks(self, user_query):
        try:
            prompt = f"Break down the following task into detailed, actionable steps:\n\nTask: {user_query}\n\nSubtasks:"
            response = generator(prompt, max_length=150)
            generated_text = response[0]['generated_text']
            
            # Post-process to extract subtasks
            subtasks = generated_text.strip().split("\n")
            subtasks = [task.strip("- ") for task in subtasks if task]  # Remove bullet points if they exist
            self.tasks = subtasks
            return subtasks
        except Exception as e:
            return [f"Error generating subtasks: {e}"]

    def modify_task(self, task_index, new_task):
        if 0 <= task_index < len(self.tasks):
            self.tasks[task_index] = new_task
        else:
            return "Invalid task index."

    def delete_task(self, task_index):
        if 0 <= task_index < len(self.tasks):
            del self.tasks[task_index]
        else:
            return "Invalid task index."

    def add_task(self, new_task):
        self.tasks.append(new_task)

    def get_subtasks(self):
        return self.tasks

# ToolAgent: Responsible for solving each subtask
class ToolAgent:
    def solve_task(self, task):
        try:
            # Let's simulate task-solving by generating a response
            prompt = f"How to solve the following task:\n\nTask: {task}\n\nSolution:"
            response = generator(prompt, max_length=100)
            solution = response[0]['generated_text']
            return solution
        except Exception as e:
            return f"Error solving the task: {e}"

# Reflection: Feedback on the task result (simulated)
def reflection(subtask_result):
    return f"Reflection on result: The solution seems {'adequate' if 'success' in subtask_result.lower() else 'inadequate'}."

# Initialize the agents
plan_agent = PlanAgent()
tool_agent = ToolAgent()

# Gradio interface functions
def break_down_task(user_input):
    subtasks = plan_agent.break_into_subtasks(user_input)
    return "\n".join(subtasks)

def solve_subtasks():
    solved_tasks = []
    subtasks = plan_agent.get_subtasks()
    for task in subtasks:
        solution = tool_agent.solve_task(task)
        solved_tasks.append(f"Task: {task}\nSolution: {solution}")
    return "\n\n".join(solved_tasks)

def modify_subtask(index, new_subtask):
    index = int(index) - 1  # Convert to zero-indexed
    result = plan_agent.modify_task(index, new_subtask)
    if result is None:
        return "\n".join(plan_agent.get_subtasks())
    else:
        return result

def add_subtask(new_task):
    plan_agent.add_task(new_task)
    return "\n".join(plan_agent.get_subtasks())

def delete_subtask(index):
    index = int(index) - 1  # Convert to zero-indexed
    result = plan_agent.delete_task(index)
    if result is None:
        return "\n".join(plan_agent.get_subtasks())
    else:
        return result

def feedback_loop():
    solved_tasks = solve_subtasks()
    reflection_results = []
    for solution in solved_tasks.split("\n\n"):
        reflection_results.append(reflection(solution))
    return "\n\n".join(reflection_results)

# Create the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("## Task Breakdown and Workflow Using PlanAgent and ToolAgent")

    # Input for task
    task_input = gr.Textbox(label="Enter a task you'd like to break into subtasks")
    
    # Button to break task into subtasks
    break_task_btn = gr.Button("Break Task into Subtasks")
    
    # Output for subtasks
    subtasks_output = gr.Textbox(label="Generated Subtasks", lines=10)

    break_task_btn.click(break_down_task, inputs=task_input, outputs=subtasks_output)
    
    # Solve subtasks
    solve_btn = gr.Button("Solve Subtasks")
    solve_output = gr.Textbox(label="Solutions to Subtasks", lines=10)
    
    solve_btn.click(solve_subtasks, outputs=solve_output)
    
    # Feedback loop for reflection
    feedback_btn = gr.Button("Reflection Feedback")
    feedback_output = gr.Textbox(label="Reflection Feedback", lines=10)
    
    feedback_btn.click(feedback_loop, outputs=feedback_output)

    # Modification input
    modify_index = gr.Number(label="Subtask Index to Modify")
    modify_input = gr.Textbox(label="New Subtask Description")
    modify_btn = gr.Button("Modify Subtask")
    modify_output = gr.Textbox(label="Subtasks after Modification", lines=10)
    
    modify_btn.click(modify_subtask, inputs=[modify_index, modify_input], outputs=modify_output)
    
    # Adding a new task
    new_task_input = gr.Textbox(label="Add a New Subtask")
    add_task_btn = gr.Button("Add Subtask")
    add_task_output = gr.Textbox(label="Subtasks after Adding", lines=10)
    
    add_task_btn.click(add_subtask, inputs=new_task_input, outputs=add_task_output)
    
    # Deleting a task
    delete_index = gr.Number(label="Subtask Index to Delete")
    delete_btn = gr.Button("Delete Subtask")
    delete_output = gr.Textbox(label="Subtasks after Deletion", lines=10)
    
    delete_btn.click(delete_subtask, inputs=delete_index, outputs=delete_output)

# Launch the Gradio app
interface.launch()
