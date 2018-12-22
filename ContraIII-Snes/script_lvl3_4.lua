--_x lives til game over


function clip(v, min, max)
    if v < min then
        return min
    elseif v > max then
        return max
    else
        return v
    end
end

prev_lives = 0
deaths_to_game_over = 4
level = data.level
function done_check()
  --print("calling done check")
  if level ~= data.level then
    return true
  end
  --end episode on death/chance to end episode on death
  if data.lives > prev_lives then
    prev_lives = data.lives
  elseif data.lives < prev_lives then
    --potential issue if you gain a life as you lose a life
    --sometimes when you gain a life it throws the count off ie have 94 lives, gain a life then have 30 in data.lives
    if prev_lives - data.lives > 5 then
      prev_lives = data.lives
      return false
    end
    prev_lives = data.lives
    deaths_to_game_over = deaths_to_game_over - 1
    if deaths_to_game_over < 0 then
      return true
    end
  end

  if data.lives <= 0 then
    return true
  end

  return false
end

prev_score = 0
death_score = -8.01 -- -1. ---1.0 * math.log(10000)
level_clear_bonus = 1000.5 --math.log(100000)
boss_clear_bonus = 100.0
boss_damage_reward = 2.0
scroll_reward = 1.000001 --math.log(10)
boss_hp_table = {}
prior_hp = 0.0
clip_min = -1100
clip_max = 1100
prev_lives_for_score = data.lives
death_frames_penalty = 0 --15
death_frames_iter = 0

--variables for scrolling in the level
x_scroll_pos = 127 --player x position that enables level to scroll
x_scroll_constant = 1 --value that tells screen is horizontal scrollable
y_scroll_constant = 3 --value that tells screen is vert scrollable
scroll_reward_steps = 0 --only scroll reward on this many of scrolls
current_scrolls = 0
score_reward = 0
scroll_multiplier = 0.0 --penalty for if level is scrollable and player is not progressing in level
existence_penalty = 0.0 ---1.0 * math.log(1.01) --life is suffering
prev_scroll_value = 0
prev_vert_scroll_value = 0
prev_scroll_constant = 1

function correct_score()
  local delta = 0
  --print('score is ' .. data.score)
  --print("about to call done check from correct score")
  if done_check() then
    --print("inside inner done check")
    if level ~= data.level then
      delta = delta + level_clear_bonus
    else
      --penalty for game over
      delta = delta + death_score
    end
    return delta
  elseif data.lives == prev_lives_for_score - 1 then
    --sometimes lose multiple lives due to weird score bug and 100 lives cheat code
    prev_lives_for_score = data.lives
    delta = delta + death_score
    return delta
  end
  prev_lives_for_score = data.lives
  --print("correct score start 1 ")



  --when scrolling, only get scroll reward, no score bonus
  --when not scrolloable only get a delta reward for points for score increase
  if data.is_scrollable == x_scroll_constant or data.is_scrollable == y_scroll_constant then
    delta = scrollable_adjustment(delta)
    --reward for beating mini boss
    --if prev_scroll_constant ~= x_scroll_constant and prev_scroll_constant ~= y_scroll_constant then
    --  delta = delta + boss_clear_bonus
      --print('giving mini boss bonus, TESTING')
    --end
  else
    if prior_hp > 0 then
      --print("testing boss hp " .. prior_hp)
      local boss_r = boss_hp_reward()
      delta = delta + boss_r
      if prior_hp == 0 then
        delta = delta + boss_clear_bonus
      end
    else
      --print("looking for boss hp values...")
      --no boss hp found, keep checking
      boss_hp_table = {}
      prior_hp = initialize_boss_hp()
    end
    --print('finding new score 1 ' .. data.score .. '  ' .. prev_score)
    --if data.score > prev_score then
      --local score_clip = clip(math.log(1+data.score-prev_score),1,5)
      --print('score clip is ' .. score_clip)
      --print('delta is ' .. delta)
      --score_clip = math.floor(score_clip+0.5) --round to nearest whole number
      --delta = delta + score_reward
      --print('finding new score 2 ' .. score_clip .. ' ' .. delta)
    --end
  end
  prev_scroll_constant = data.is_scrollable

  --penalty for death lasts for death_frames_penalty frames
  --if death_frames_iter > 0 then
  --  death_frames_iter = death_frames_iter - 1
  --  delta = delta + death_score
  --end

  --not sure if score ever resets, never really figured out the rom address to score calculation
  prev_score = data.score

  delta = clip(delta, clip_min, clip_max)
  --print('what is delta? ' .. delta)
  return delta
end

--get scroll reward for being on part of screen that scrolls but not sitting there, ie get the reward when the data.scroll_value changes
function scrollable_adjustment(r)
  
  if data.is_scrollable == x_scroll_constant then
    if data.scroll_value ~= prev_scroll_value then
      prev_scroll_value = data.scroll_value
      current_scrolls = current_scrolls + 1
      if current_scrolls >= scroll_reward_steps then
        r = r + scroll_reward
        current_scrolls = 0
      end
    --else
      --user position >= 127 to scroll
      --local distance_penalty = 0.
      --if data.x1 < x_scroll_pos then
      --  distance_penalty = -1 * math.log(1 + math.abs(x_scroll_pos-data.x1)/127.0) 
      --end
      --r = r + existence_penalty
      --r = clip(r,clip_min,scroll_reward*.5)--r -- scroll_multiplier --+ existence_penalty --+ distance_penalty
    end
  elseif data.is_scrollable == y_scroll_constant then
    if data.vert_scroll_value ~= prev_vert_scroll_value then
      --user data.x1 < 89 then goes up (but toggles between 89 and 90 as it goes up)
      prev_vert_scroll_value = data.vert_scroll_value
      if current_scrolls >= scroll_reward_steps then
        r = r + scroll_reward
        current_scrolls = 0
      end
    --else
      --r = r + existence_penalty
      --r = clip(r,clip_min,scroll_reward*.5) --r * scroll_multiplier --+ existence_penalty
    end
  else
    --print("in scrollable adjustment " .. r)
    r = r --+ existence_penalty
  end
  return r
end

--when scrolling stops, checks for boss hp and sets prior_hp to value found
--also makes a table of which HP values are active so can do reward for dmg
--when boss hp is found, stops doing this check
function initialize_boss_hp()
  local hp_string = "hp_"
  local hp_min = 16
  local hp_found = 0
  for i = 0,32,1
  do
    local hp_key = hp_string .. i
    if data[hp_key] > hp_min then
      hp_found = hp_found + data[hp_key]
      boss_hp_table[hp_key] = hp_key
    end
  end

  return hp_found
end

--relies on global variables boss prior hp
--list of dict keys to get current hp
function boss_hp_reward()
  local current_hp = 0
  for k, v in pairs(boss_hp_table) do
    --print("finding hp amount at data part " .. k .. " " .. data[k])
    if data[k] > 0 then
      current_hp = current_hp + data[k]
    end
  end
  r = (prior_hp - current_hp) * boss_damage_reward
  --print("giving reward for boss dmg " .. r )
  r = clip(r,0,clip_max)
  prior_hp = current_hp
  return r
end
