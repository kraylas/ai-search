(ns search.core)

(require '[clojure.set :refer [union]])

(defn get-changed-idx [idx]
  (case idx
    0 [1, 3]
    1 [0, 2, 4]
    2 [1, 5]
    3 [0, 4, 6]
    4 [3, 1, 5, 7]
    5 [4, 2, 8]
    6 [3, 7]
    7 [6, 4, 8]
    8 [7, 5]
    (println (str "default: wrong input :" idx))))

(defn get-new-state [st]
  (let [idx (.indexOf st 0) n (get-changed-idx idx) vid (nth st idx)]
    (map (fn [i] (-> st (assoc idx (nth st i)) (assoc i vid))) n)))

(defn vec2set [v] (apply union (map (fn [i] #{i}) v)))

(defn bfs [init-state target-state is-random]
  (loop [q [init-state] h #{init-state}]
    (let [x (first q)] ; 有趣的是，仅仅把队列的出入顺序改一下，bfs就变成了dfs
      (if (= x target-state)
        (println (str "search successful, state count:" (count h)))
        (let [rs (vec (filter (fn [st] (not (contains? h st))) (get-new-state x)))]
          (recur (vec (apply conj (rest q) (if is-random (shuffle rs) rs))) (union h (vec2set rs) #{x})))))))

(defn dfs [init-state target-state is-random]
  (loop [q [init-state] h #{init-state}]
    (let [x (peek q)]
      (if (= x target-state)
        (println (str "search successful, state count:" (count h)))
        (let [rs (vec (filter (fn [st] (not (contains? h st))) (get-new-state x)))]
          (recur (vec (apply conj (pop q) (if is-random (shuffle rs) rs))) (union h (vec2set rs) #{x})))))))


(defn -main [] 
  (println "dfs non-random:")
  (time (dfs [2 8 3 1 6 4 7 0 5] [1 2 3 8 0 4 7 6 5] false))
  (println "bfs non-random:")
  (time (bfs [2 8 3 1 6 4 7 0 5] [1 2 3 8 0 4 7 6 5] false))

  (println "dfs random 10 rounds:")

  (vec (for [_ (range 10)] ; prevent lazy evaluation
         (let []
           (println (str _ "th:"))
           (time (dfs [2 8 3 1 6 4 7 0 5] [1 2 3 8 0 4 7 6 5] true)))))

  (println "bfs random 10 rounds:")

  (vec (for [_ (range 10)]
         (let []
           (println (str _ "th"))
           (time (bfs [2 8 3 1 6 4 7 0 5] [1 2 3 8 0 4 7 6 5] true))))))
