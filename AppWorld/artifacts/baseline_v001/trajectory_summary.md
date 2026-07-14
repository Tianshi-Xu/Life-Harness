# AppWorld Qwen3-4B Baseline Trajectory Analysis

This report uses runtime trajectories and final success booleans only. It does not read solutions, private oracle data, or evaluation requirement text.

## Aggregate

| Metric | All | Failed | Successful |
|---|---:|---:|---:|
| tasks | 90 | 81 | 9 |
| lm_calls | 806 | 730 | 76 |
| action_count | 847 | 764 | 83 |
| tasks_with_environment_errors | 48 | 45 | 3 |
| environment_error_count | 175 | 172 | 3 |
| no_tool_turns | 89 | 85 | 4 |
| missing_required_fields | 1 | 1 | 0 |
| unknown_argument_fields | 7 | 7 | 0 |
| string_integer_fields | 0 | 0 | 0 |
| tasks_with_duplicate_actions | 7 | 7 | 0 |
| consecutive_duplicate_actions | 51 | 51 | 0 |
| tasks_with_abab_loops | 3 | 3 | 0 |
| tasks_with_fail_submission | 30 | 30 | 0 |
| missing_completion | 7 | 7 | 0 |
| budget_exhausted | 0 | 0 | 0 |
| single_page_exhaustive_candidates | 42 | 39 | 3 |

## Environment error types

- `traceback`: 175
- `validation`: 3
- `state_conflict`: 3
- `not_found`: 1
- `authentication`: 1
- `invalid_value`: 1

## Environment error fingerprints

- `phone__login | {"message":"Invalid credentials"}`: 52
- `file_system__show_file | Exception: Response status code is <N>:`: 42
- `file_system__show_directory | {"message":"You are either not authorized to access this file_system API endpoint or your access token is missing, invalid or expired."}`: 8
- `phone__show_contacts | NameError: name 'phone__show_contacts' is not defined`: 6
- `spotify__play_music | {"message":"You are either not authorized to access this spotify API endpoint or your access token is missing, invalid or expired."}`: 6
- `phone__search_contacts | {"message":"You are either not authorized to access this phone API endpoint or your access token is missing, invalid or expired."}`: 5
- `simple_note__search_notes | {"message":"You are either not authorized to access this simple_note API endpoint or your access token is missing, invalid or expired."}`: 4
- `spotify__show_song | Exception: Unexpected parameter 'access_token' passed to the show_song API of the spotify app. Allowed parameters are: ['song_id']`: 3
- `phone__show_alarms | {"message":"You are either not authorized to access this phone API endpoint or your access token is missing, invalid or expired."}`: 3
- `phone__get_contact_info | NameError: name 'phone__get_contact_info' is not defined`: 3
- `spotify__follow_artist | Exception: Response status code is <N>:`: 3
- `spotify__show_downloaded_songs | {"message":"You are either not authorized to access this spotify API endpoint or your access token is missing, invalid or expired."}`: 2
- `spotify__login | {"message":"Invalid credentials"}`: 2
- `venmo__show_social_feed | {"message":"You are either not authorized to access this venmo API endpoint or your access token is missing, invalid or expired."}`: 2
- `file_system__directory_exists | {"message":"You are either not authorized to access this file_system API endpoint or your access token is missing, invalid or expired."}`: 2
- `phone__login | Exception: Unexpected parameter 'phone_number' passed to the login API of the phone app. Allowed parameters are: ['username', 'password']`: 2
- `spotify__review_song | Exception: Response status code is <N>:`: 2
- `venmo__show_transactions | {"message":"You are either not authorized to access this venmo API endpoint or your access token is missing, invalid or expired."}`: 2
- `spotify__remove_song_from_library | Exception: Response status code is <N>:`: 2
- `simple_note__update_note | {"message":"You are either not authorized to access this simple_note API endpoint or your access token is missing, invalid or expired."}`: 2
- `simple_note__login | Exception: Response status code is <N>:`: 1
- `venmo__create_payment_request | {"message":["The user with email <EMAIL> does not exist."]}`: 1
- `spotify__show_album | Exception: Unexpected parameter 'access_token' passed to the show_album API of the spotify app. Allowed parameters are: ['album_id']`: 1
- `spotify__search_songs | Exception: Response status code is <N>:`: 1
- `phone__send_text_message | {"message":"You are either not authorized to access this phone API endpoint or your access token is missing, invalid or expired."}`: 1
- `venmo__show_social_feed | NameError: name 'phone__get_contacts' is not defined`: 1
- `phone__login | Exception: Response status code is <N>:`: 1
- `phone__get_contacts | NameError: name 'phone__get_contacts' is not defined`: 1
- `phone__get_current_date_and_time | Exception: Unexpected parameter 'access_token' passed to the get_current_date_and_time API of the phone app. Allowed parameters are: []`: 1
- `supervisor__complete_task | Exception: Unexpected parameter 'message' passed to the complete_task API of the supervisor app. Allowed parameters are: ['answer', 'status']`: 1
- `file_system__create_directory | {"message":"You are either not authorized to access this file_system API endpoint or your access token is missing, invalid or expired."}`: 1
- `file_system__show_file | {"message":"You are either not authorized to access this file_system API endpoint or your access token is missing, invalid or expired."}`: 1
- `file_system__login | Exception: Response status code is <N>:`: 1
- `supervisor__show_payment_cards | {"message":"You are either not authorized to access this phone API endpoint or your access token is missing, invalid or expired."}`: 1
- `spotify__show_song_library | Exception: Response status code is <N>:`: 1
- `spotify__show_album_library | {"message":"You are either not authorized to access this spotify API endpoint or your access token is missing, invalid or expired."}`: 1
- `spotify__current_user_followed_artists | NameError: name 'spotify__current_user_followed_artists' is not defined`: 1
- `venmo__login | {"message":"Invalid credentials"}`: 1
- `spotify__complete_task | NameError: name 'spotify__complete_task' is not defined`: 1
- `spotify__like_song | Exception: Response status code is <N>:`: 1
- `spotify__show_song_library | Exception: Unexpected parameter 'min_release_date' passed to the show_song_library API of the spotify app. Allowed parameters are: ['access_token', 'query', 'page_index', 'page_limit', 'sort_by']`: 1
- `spotify__show_playlist_library | Exception: Response status code is <N>:`: 1

## Schema issues

- `spotify__show_song:unknown:access_token`: 3
- `spotify__show_album:unknown:access_token`: 1
- `supervisor__complete_task:unknown:message`: 1
- `spotify__show_song_library:unknown:max_release_date`: 1
- `spotify__show_song_library:unknown:min_release_date`: 1
- `simple_note__search_notes:missing:access_token`: 1

## Most common actions

- `supervisor__show_account_passwords`: 85
- `supervisor__complete_task`: 84
- `spotify__show_song`: 73
- `phone__login`: 59
- `file_system__show_file`: 48
- `spotify__login`: 43
- `supervisor__show_profile`: 41
- `spotify__like_song`: 41
- `spotify__show_playlist`: 26
- `spotify__add_song_to_playlist`: 20
- `spotify__follow_artist`: 19
- `venmo__login`: 19
- `spotify__review_song`: 17
- `simple_note__search_notes`: 14
- `venmo__show_transactions`: 14
- `phone__search_contacts`: 13
- `phone__delete_voice_message`: 13
- `spotify__show_song_library`: 11
- `phone__get_current_date_and_time`: 11
- `simple_note__login`: 10
- `simple_note__show_note`: 10
- `venmo__like_transaction`: 10
- `file_system__show_directory`: 10
- `spotify__show_playlist_library`: 10
- `venmo__approve_payment_request`: 10
- `phone__delete_text_message`: 10
- `spotify__remove_song_from_library`: 10
- `spotify__show_liked_songs`: 9
- `spotify__search_songs`: 9
- `spotify__search_artists`: 8
