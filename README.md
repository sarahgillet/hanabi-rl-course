# hanabi-rl-course
Some pipelining with hanabi and DRL for a course

Install requirements with
```
pip install -r requirements.txt
```

Set up environment variables:
 ```
 cp .env_template .env
 vim .env`
 ```

Locate the checkpoints from training.


Run with:
```
python -um main \
  --base_dir=absolute_path_to_directory_containing_checkpoints_folder \
  --gin_files='configs/hanabi_rainbow.gin'
```
In a browser, log on to Hanabi Live and start a new table.
* In the pre-game chat window, send a private message to the bot in order to get it to join you (version without password):
  * `/msg [username] /join`
* If you have set a password to the game send the following message instead
  * `/msg [username] /join [yourPassword]`
Then, start the game and play!
