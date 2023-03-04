import os
import discord
from discord.ext import commands
from discord import Client, Embed, Thread
from discord.ui import View, Button
from discord.interactions import InteractionType
import openai
import asyncio
import uuid
import json
import numpy as np
from summa import keywords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

# Create a new Discord bot client with the appropriate intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = False
intents.voice_states = False
intents.presences = False
#client = commands.Bot(command_prefix="/", intents=intents)

client = discord.Client(intents=intents)
# Set up the OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]
# Set the base URL for the API
openai.api_base = "https://api.openai.com/v1/chat"

allowed_channels = config['GENERAL']['help_channel']
threads_file = 'threads.json'


def generate_response(conversation_history):
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        temperature=0.7,
        max_tokens=1024,
        n=1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    message = response.choices[0]["message"]["content"].strip()

    print(f'\nGenerated Response:\n{message}\n')

    return message

def load_threads():
    if os.path.exists(threads_file):
        with open(threads_file, 'r') as f:
            try:
                threads = json.load(f)
            except json.JSONDecodeError:
                threads = {}
    else:
        threads = {}
    return threads

def save_threads(threads):
    with open(threads_file, 'w') as f:
        json.dump(threads, f)

async def get_thread_history(thread):
    threads = load_threads()
    thread_id = str(thread.id)
    if thread_id in threads:
        x = threads[thread_id]['messages']
        return threads[thread_id]['messages']
    else:
        history = []
        threads[thread_id] = {'messages': history}
        save_threads(threads)
        return history

async def join_threads():
    threads = load_threads()
    for thread_id, thread_data in threads.items():
        try:
            thread = await client.fetch_channel(thread_id)
            get_thread_history(thread)
            if thread.parent_id in allowed_channels:
                await thread.join()
        except:
            pass

# Load the keywords from the JSON file
with open("keywords.json", "r") as f:
    keywords_dict = json.load(f)

# Keywords
vectorizer = TfidfVectorizer()

def find_matching_file(words):
    """
    Finds a file in the context directory that matches the given keywords.
    """
    matching_file = None
    best_similarity = 0

    context_dir = "context"
    for filename in os.listdir(context_dir):
        file_path = os.path.join(context_dir, filename)
        with open(file_path, "r") as file:
            file_content = file.read()
            file_keywords = keywords_dict.get(filename, [])
            similarity = compute_cosine_similarity(words, file_keywords)
            #print(f'\nFile:\n{filename}\nKeyword Simularity:\n{similarity}')
            if similarity > best_similarity:
                best_similarity = similarity
                matching_file = file_path

    return matching_file

def compute_cosine_similarity(set1, set2):
    """
    Computes the cosine similarity between two sets of text data.
    """
    # Convert the sets to strings
    str1 = " ".join(list(set1))
    str2 = " ".join(list(set2))

    # Create a CountVectorizer object
    tfidf_matrix = vectorizer.fit_transform([str1, str2])

    # Compute the cosine similarity
    similarity = cosine_similarity(tfidf_matrix)[0][1]

    return similarity

class CircularBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = [None] * max_size
        self.head = 0
        self.tail = 0
        self.size = 0

    def append(self, item, thread, role):
        self.buffer[self.head] = item
        self.head = (self.head + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1
        elif self.head == self.tail:
            self.tail = (self.tail + 1) % self.max_size

        # Write buffer to thread message history
        self.write(thread, role, item)

    def get(self, thread):
        # Load the existing threads.json file
        with open('threads.json', 'r') as f:
            threads = json.load(f)
        messages = threads[str(thread.id)]['messages']
        return messages

    def write(self, thread, role, content):
        thread_id = str(thread.id)
        # Load the existing threads.json file
        with open('threads.json', 'r') as f:
            threads = json.load(f)
        # If the thread ID is not in the threads dictionary, add it
        if thread_id not in threads:
            threads[thread_id] = {'messages': []}
        # Append the new message to the thread's message list
        threads[thread_id]['messages'].append({'role': role, 'content': content})
        # If the message list is longer than the max size, remove the oldest message
        max_size = self.max_size
        if len(threads[thread_id]['messages']) > max_size:
            threads[thread_id]['messages'].pop(0)
        # Write the updated threads dictionary back to the threads.json file
        with open('threads.json', 'w') as f:
            json.dump(threads, f, indent=4)

    def delete_oldest_messages(self, thread, max_size):
        # Load the existing threads.json file
        with open('threads.json', 'r') as f:
            threads = json.load(f)
    
        # Get the messages for the specified thread
        messages = threads[str(thread.id)]['messages']
    
        # Calculate how many messages to delete
        num_messages_to_delete = len(messages) - max_size
    
        if num_messages_to_delete <= 0:
            # There are not enough messages to delete
            return
    
        # Delete the oldest messages
        messages_to_delete = messages[:num_messages_to_delete]
        messages = messages[num_messages_to_delete:]
    
        # Put the updated messages back in the threads dictionary
        threads[str(thread.id)]['messages'] = messages
    
        # Write the updated threads dictionary back to the threads.json file
        with open('threads.json', 'w') as f:
            json.dump(threads, f, indent=4)
    
        # Print a message to indicate which messages were deleted
        print(f"Deleted {num_messages_to_delete} messages from thread {thread.id}: {messages_to_delete}")

def split_message(message):
    chunks = []
    while len(message) > 0:
        if len(message) <= 1500:
            chunks.append(message)
            break
        split_index = message.rfind(' ', 0, 1500)
        chunks.append(message[:split_index])
        message = message[split_index+1:]
    return chunks

conversation_history = CircularBuffer(max_size=5)

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    print(f'\nUsers Message:\n{message.content}\n')

    if isinstance(message.channel, Thread):
        thread = message.channel

        await get_thread_history(thread)

        # Split the input message into individual words
        words = message.content.split()

        # Find the matching file based on the words
        matching_file = find_matching_file(words)
        if matching_file is not None:
            with open(matching_file, "r") as f:
                file_content = f.read()
            conversation_history.append(file_content , thread, 'system')

        print(f"\nMatching file:\n{matching_file}\n")

        conversation_history.append(message.content, thread, 'user')

        max_size = 50

        while True:
            # Generate a response
            try:
                response = generate_response(conversation_history.get(thread))
                break
            except openai.error.InvalidRequestError:
                max_size = max_size // 2
                # If the response cannot be generated due to maximum context length limit, remove the oldest messages and try again
                conversation_history.delete_oldest_messages(thread, max_size)

        conversation_history.append(response, thread, 'assistant')
        try: 
            await thread.send(response)
        except discord.errors.HTTPException:
            chunks = split_message(response)
            for chunk in chunks:
                await message.channel.send(chunk)


class Chat(discord.ui.View):
    @discord.ui.button(label="Chat now!", style=discord.ButtonStyle.primary) # Create a button with the label "😎 Click me!" with color Blurple
    async def button_callback(self, button, interaction):
        # Get the user who clicked the button
        user = interaction.user

        # Generate an 8-character ID for the chat thread
        chat_id = str(uuid.uuid4())[:8]
        thread_name = f"Chat - {chat_id}"

        # Create a new thread with the thread name and add the user to it
        thread = await interaction.channel.create_thread(name=thread_name, auto_archive_duration=60)
        await thread.add_user(user)

        conversation_history.append('You are ChatDSA, you are a helpful assistant for Royal Productions', thread, 'system')

        # Send a welcome message to the user in the thread
        await thread.send(f"Hello, {user.mention}! How can I help you today?")

@client.event
async def on_ready():
    print(f'{client.user} is ready')
    channel = await client.fetch_channel(allowed_channels) # Replace with the channel ID you want to send the message to
    
    embed = Embed(
        title="Need Help? Chat with ChatDSA",
        description="By clicking the button below, you can talk to ChatDSA for help",
        color=0x7851a9, # Replace with your desired color (in decimal format)
    )
    embed.set_footer(text="ChatDSA by Royal Productions", icon_url=f"https://raw.githubusercontent.com/ihasTaco/ChatDSA/main/ChatDSA.png")
    view = Chat()

    if config['GENERAL']['message_id'] == '0':
        message = await channel.send(embed=embed, view=Chat())

        # Update the message_id value
        config.set('GENERAL', 'message_id', str(message.id))

        # Save the changes to the config file
        with open('config.ini', 'w') as f:
            config.write(f)
    else:
        try:
            message = await channel.fetch_message(config['GENERAL']['message_id'])

            # Add the view to the message
            await message.edit(view=view)
        except discord.NotFound:
            message = await channel.send(embed=embed, view=Chat())

            # Update the message_id value
            config.set('GENERAL', 'message_id', str(message.id))

            # Save the changes to the config file
            with open('config.ini', 'w') as f:
                config.write(f)

client.run("MTA4MDU2NTk2ODk0MzcxMDI0OQ.Gorcoi.10Lz39c2KJi-kT-wRKVSCWaVqKfS3OAcyj5RSY")

if __name__ == '__main__':
    asyncio.run(join_threads())
