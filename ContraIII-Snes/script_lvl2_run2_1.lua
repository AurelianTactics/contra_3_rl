-- _x lives til game over
--print("loading lua script")
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
deaths_to_game_over = 1
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

--get reward for
--killing boss
--clearing level
--getting to boss
--causing dmg to a pod
--destroying a pod
--getting close to next pod

is_boss = 0 --in boss part of level or pre boss part
pod_health = 0 --stores health of all pods
pods_remaining = 0 --number of pods remaining
is_pod_health_set = 0
pod_hp_table = {}
pod_hp_table['pod_0'] = 32
pod_hp_table['pod_1'] = 32
pod_hp_table['pod_2'] = 32
pod_hp_table['pod_3'] = 32
pod_hp_table['pod_4'] = 32
current_target_pod = 0

pod_health_reward = 10.000001
pod_clear_reward = 100.0
boss_health_reward = 10.0
death_reward = -0.01
pod_distance_reward = 1.0
boss_clear_bonus = 500.0
level_clear_bonus = 1000.5
clip_min = -1100
clip_max = 1100
prev_boss_health = -1 --not ideal but have to initialize it, gives a fake reward for boss start

prev_lives_for_score = data.lives

--initialize distance far away, first time looking for that pod set distance to actual distance
distance_start = 3000
pod_distance_table = {}
pod_distance_table['pod_0'] = distance_start
pod_distance_table['pod_1'] = distance_start
pod_distance_table['pod_2'] = distance_start
pod_distance_table['pod_3'] = distance_start
pod_distance_table['pod_4'] = distance_start

--on destroying a pod, distances can flicker
function reset_pod_distance_table()
  pod_distance_table['pod_0'] = distance_start
  pod_distance_table['pod_1'] = distance_start
  pod_distance_table['pod_2'] = distance_start
  pod_distance_table['pod_3'] = distance_start
  pod_distance_table['pod_4'] = distance_start
end

function correct_score()
  local delta = 0
  --print("correct score start 0")
  if done_check() then
    --print("inside inner done check")
    if level ~= data.level then
      delta = delta + level_clear_bonus
    else
      --penalty for game over
      delta = delta + death_reward
    end
    return delta
  elseif data.lives == prev_lives_for_score - 1 then
    --sometimes lose multiple lives due to weird score bug and 100 lives cheat code
    prev_lives_for_score = data.lives
    delta = delta + death_reward
    return delta
  end
  prev_lives_for_score = data.lives
  --print("correct score start 1 ")
  --initialize pod healths when level loads
  if is_pod_health_set == 0 then
    local pod_temp = initialize_pod_hp()
    if pod_temp == 32 * 5 then
      is_pod_health_set = 1
      pod_health = 32 * 5
      pods_remaining = 5
      current_target_pod = get_closest_target_pod()
    end
  end
  --print("correct score start 2 ")
  if is_boss == 0 then
    --pre boss part, get reward for doing dmg to pods, kiling pods, getting close to target pod
    if is_pod_health_set == 1 then 
      --reward for damaging a pod
      delta = delta + get_pod_hp_reward()
      --reward for destroying a pod
      local current_pods = get_pods_remaining()
      if current_pods < pods_remaining then
        pods_remaining = current_pods
        delta = delta + pod_clear_reward
        current_target_pod = get_closest_target_pod()
      end
      --reward for getting closer to pods
      if pods_remaining > 0 then
        delta = delta + get_pod_distance_bonus()
      end
      
      if pod_health == 0 then
        is_boss = 1
        delta = delta + boss_clear_bonus
      end
    end
  else
    --boss part, get damage for damaging boss
    local current_boss_health = 0
    if data.level == 2 then
      current_boss_health = data.pod_2
    else
      current_boss_health = data.pod_0
    end
    --boss health loops, if health jumps up then it looped, if not take difference
    if current_boss_health > prev_boss_health then
      delta = delta + boss_health_reward
    else
      delta = delta + (prev_boss_health - current_boss_health) * boss_health_reward
    end
    prev_boss_health = current_boss_health
  end
  delta = clip(delta,clip_min,clip_max)
  --print("correct score end")
  return delta
end

--sets the target pod to go for next
function get_random_target_pod()
  math.randomseed(data.score+1)
  local tgt = math.random(0,4)
  --print(" pod target is " .. tgt )
  local pod_key = "pod_" .. tgt
  local loop_count = 0
  
  
  while(loop_count < 5)
  do
    --if target found then break
    if data[pod_key] > 0 and data[pod_key] < 33 then
      break
    end
    loop_count = loop_count + 1
    tgt = (tgt + 1) % 5
    pod_key = "pod_" .. tgt
    --print("testing tgt finder " .. tgt )
  end
  --print("getting random pod target " .. tgt)
  reset_pod_distance_table()
  return tgt
end

function initialize_pod_hp()
  --print("intialize pod hp start")
  local pod_string = "pod_"
  local pod_start = 32
  local pod_temp = 0
  for i = 0,4,1
  do
    local pod_key = pod_string .. i
    if data[pod_key] == pod_start then
      pod_temp = pod_temp + pod_start
    end
  end
  --print("intialize pod hp end")
  return pod_temp
  --if pod_temp == pod_start * 5 then
  --  return pod_start * 5
  --end
  --return 0
end

function get_pod_hp_reward()
  --print("get pod hp start")
  local current_hp = 0
  for k, v in pairs(pod_hp_table) do
    --print("finding hp amount at data part " .. k .. " " .. data[k])
    if data[k] > 0 and data[k] < 33 then
      current_hp = current_hp + data[k]
    end
  end

  r = (pod_health - current_hp) * pod_health_reward
  --print("giving reward for pd dmg " .. r )
  r = clip(r,0,clip_max)
  pod_health = current_hp
  --if r > 0 then
  --  print("returning reward for pod hp dmg " .. r)
  --end
  
  return r
end



--track number of pods with 1 to 32 health
function get_pods_remaining()
  --print("get pod hp start")
  local num_pods = 0
  for k, v in pairs(pod_hp_table) do
    --print("finding hp amount at data part " .. k .. " " .. data[k])
    if data[k] > 0 and data[k] < 33 then
      num_pods = num_pods + 1
    end
  end
  --print("number of pods remaining is " .. num_pods)
  return num_pods
end

function get_exact(precise,rough)
  local x = precise + rough * 255.0 
  return x
end



pod_x_table = {}
pod_y_table = {}

--coordinates of pds for lvl 2 and lvl 5
if data.level == 2 then
  pod_x_table['pod_0'] = get_exact(148.0,0.0)
  pod_x_table['pod_1'] = get_exact(112.0,3.0)
  pod_x_table['pod_2'] = get_exact(114.0,1.0)
  pod_x_table['pod_3'] = get_exact(148.0,2.0)
  pod_x_table['pod_4'] = get_exact(116.0,0.0)
  pod_y_table['pod_0'] = get_exact(112.0,0.0)
  pod_y_table['pod_1'] = get_exact(80.0,0.0)
  pod_y_table['pod_2'] = get_exact(142.0,2.0)
  pod_y_table['pod_3'] = get_exact(146.0,2.0)
  pod_y_table['pod_4'] = get_exact(142.0,3.0)
else
  pod_x_table['pod_0'] = get_exact(181.0,0.0)
  pod_x_table['pod_1'] = get_exact(111.0,2.0)
  pod_x_table['pod_2'] = get_exact(145.0,3.0)
  pod_x_table['pod_3'] = get_exact(83.0,3.0)
  pod_x_table['pod_4'] = get_exact(245.0,0.0)
  pod_y_table['pod_0'] = get_exact(147.0,0.0)
  pod_y_table['pod_1'] = get_exact(113.0,1.0)
  pod_y_table['pod_2'] = get_exact(14.0,1.0)
  pod_y_table['pod_3'] = get_exact(239.0,2.0)
  pod_y_table['pod_4'] = get_exact(52.0,3.0)
end

function get_closest_target_pod()
  local current_x = get_exact(data.precise_x, data.rough_x)
  local current_y = get_exact(data.precise_y, data.rough_y)
  local tgt = 0
  local min_distance = 191919
  for i = 0,4,1
  do
    local pod_key = "pod_" .. i
    if data[pod_key] > 0 and data[pod_key] < 33 then
      local current_dist = math.abs(pod_x_table[pod_key] - current_x) + math.abs(pod_y_table[pod_key] - current_y)
      if current_dist < min_distance then
        tgt = i
        min_distance = current_dist 
      end
    end
  end
  --print("getting closest target pod " .. tgt)
  return tgt
end


--rewarded for setting a new best closeness to individual pod that remands
--at initialization and on a pod being destroyed, randomly choose a new current_target_pod
--only receive distance reward for approaching current target pod
--alternatives could be: penalty/punishment for getting to closest, euclidean instead of manhattan distance, penalty/punishment for getting close to more than one, dynamic shifting between which pod to go to

pod_min_closeness = 100
--distance checks are unreliable can flicker on a single frame, can't get distance reward for moving too much
distance_threshold = 15

function get_pod_distance_bonus()
  local distance_reward = 0

  if data.rough_x > 3 or data.rough_x < 0 or data.rough_y > 3 or data.rough_y < 0 then
    distance_frame_check = 0
    return distance_reward
  end
  local current_x = get_exact(data.precise_x, data.rough_x)
  local current_y = get_exact(data.precise_y, data.rough_y)

  for k, v in pairs(pod_hp_table) do
    if k == 'pod_' .. current_target_pod then

      if pod_distance_table[k] == distance_start then
        --initialize the starting distance from player to the pods
        local current_dist = math.abs(pod_x_table[k] - current_x) + math.abs(pod_y_table[k] - current_y)
        if current_dist < pod_min_closeness then
          pod_distance_table[k] = 0
        else
          pod_distance_table[k] = current_dist
        end
        --print('initializing current distance for first time ' .. current_dist )
      elseif pod_distance_table[k] > 0 then
        --print("finding pod distance x,y " .. current_x .. "," .. current_y)
        --print("rough x and y are " .. data.rough_x .. " " .. data.rough_y)
        if data[k] > 0 and data[k] < 33 then
          local current_dist = math.abs(pod_x_table[k] - current_x) + math.abs(pod_y_table[k] - current_y)
          --print('current distance is ' .. current_dist .. ' max distance from before is ' .. pod_distance_table[k] .. ' tgt ' .. current_target_pod ) 
          if current_dist < pod_distance_table[k] then
            if pod_distance_table[k] - current_dist <= distance_threshold then
              distance_reward = distance_reward + (pod_distance_table[k] - current_dist)*pod_distance_reward
              pod_distance_table[k] = current_dist
            end
          end

          if current_dist < pod_min_closeness then
            --reached close enough to pod, no longer need the reward
            pod_distance_table[k] = 0
          end
        else
          pod_distance_table[k] = 0 --pod destroyed, no longer getting reward for this pod
        end
      end

    end
  end

  distance_reward = clip(distance_reward,0,clip_max)
  --if distance_reward > 0 then
  --  print("getting distance_reward " .. distance_reward .. " tgt is " .. current_target_pod)
  --end

  return distance_reward
end
--[=====[ 
--]=====]


