package cn.code.utils;

import org.springframework.util.CollectionUtils;

import java.math.BigDecimal;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * @author qi.shen
 * @create 2019-06-17 15:23
 */
public class Test {

    public class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
  }



   public class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode() {}
      TreeNode(int val) { this.val = val; }
      TreeNode(int val, TreeNode left, TreeNode right) {
          this.val = val;
          this.left = left;
          this.right = right;
      }
  }

  private static ArrayBlockingQueue<Integer> queue = new ArrayBlockingQueue<Integer>(10);

    // private static LinkedBlockingQueue<Integer> queue = new LinkedBlockingQueue<>();

     static class Product implements Runnable{

         private ArrayBlockingQueue<Integer> queue;

         public Product(ArrayBlockingQueue<Integer> queue){
             this.queue = queue;
         }

          @Override
         public void run(){
              try {
                  Random random = new Random();
                  queue.put(random.nextInt());
                  System.out.println("生产" + Thread.currentThread().getName() + random.nextInt());
              }catch (Exception e){
                  e.printStackTrace();
              }
          }

     }

     static class Consumer implements Runnable{
         private ArrayBlockingQueue<Integer> queue;

         public Consumer(ArrayBlockingQueue<Integer> queue){
             this.queue = queue;
         }

         @Override
         public void run(){
             try {
                 Integer integer = queue.take();
                 System.out.println("消费" + Thread.currentThread().getName() + integer);
             }catch (Exception e){
                 e.printStackTrace();
             }
         }
     }




    public static void main(String[] agrs) {

        // int[] nums1 = new int[]{1,2};
        // int[] nums2 = new int[]{3,4};
        // double d = findMedianSortedArrays(nums1,nums2);

        // int aaa = uniquePaths(23,12);
        // System.out.println(aaa);

        // boolean a = wordBreak("applepenapple", Arrays.asList(new String[]{"apple", "pen"}));
        // char[][] chars = {{'1','1','0','0','0'},{'1','1','0','0','0'},{'0','0','1','0','0'},{'0','0','0','1','1'}};
        // int[][] prerequisites = {{1,4,7,11,15},{2,5,8,12,19},{3,6,9,16,22},{10,13,14,17,24},{18,21,23,26,30}};
        // int a =numIslands(chars) ;
        // int bbb [] = new int[]{2,3,4};

        // List<int[]> ans = new ArrayList<int[]>();
        // ans.add(1,bbb);
        // String a = decodeString("3[a2[c]]");
        // List<Integer> llll = findAnagrams("aa","bb");
        // int a = countSubstrings("aaaaa");
        // System.out.println(a);

        ExecutorService service = Executors.newFixedThreadPool(15);
        for (int i = 0;i<3;i++){
            service.execute(new Product(queue));
        }

        for (int i = 0;i<10;i++){
            service.execute(new Consumer(queue));
        }

    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        back2222(res,nums,0,cur);
        return res;
    }

    private void back2222(List<List<Integer>> res,int[] nums,int index,List<Integer> cur){
         if (index == nums.length){
             res.add(cur);
             return;
         }
        List<Integer> cur1 = new ArrayList<>();
        List<Integer> cur2 = new ArrayList<>();
        cur1.addAll(cur);
        cur2.addAll(cur);
         back2222(res,nums,index +1,cur1);
        cur2.add(nums[index]);
         back2222(res,nums,index +1,cur2);
    }



    public boolean findWhetherExistsPath(int n, int[][] graph, int start, int target) {
        ArrayList<ArrayList<Integer>> list = new ArrayList<ArrayList<Integer>>(n);
        for (int i = 0 ;i <n;i++){
            list.add(new ArrayList<>());
        }
        for (int i = 0; i<graph.length;i++){
            list.get(graph[i][0]).add(graph[i][1]);
        }
        Queue<Integer> queue = new LinkedList<>();
        boolean [] visited = new boolean[n];
        queue.add(start);
        while (!queue.isEmpty()){
            int cur = queue.poll();
            for (int j = 0;j<list.get(cur).size();j++){
                int node = list.get(cur).get(j);
                if (!visited[node]){
                    visited[node] = true;
                    if (node == target){
                        return true;
                    }
                    queue.add(node);
                }
            }
        }
        return false;
    }

        public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null){
            return res;
        }

        List<TreeNode> oneNodeList = new ArrayList<>();
        oneNodeList.add(root);
        fff(oneNodeList,res);
        return res;
    }

    private void fff(List<TreeNode> list,List<List<Integer>> res){
        List<Integer> curList = new ArrayList<>();
        List<TreeNode> nextList = new ArrayList<>();
        if (list.size() == 0){
            return;
        }
        for (TreeNode root : list){
            if (root == null){
                continue;
            }
            curList.add(root.val);
            if (root.left != null){
                nextList .add(root.left);
            }
            if (root.right != null){
                nextList .add(root.right);
            }

        }
        res.add(curList);
        fff(nextList,res);

    }



    int depth = 0;
    public int maxDepth(TreeNode root) {
        if (root == null){
            return depth;
        }
        int cur = 0;
        aaaaaaa(root,cur);
        return depth;
    }

    private void aaaaaaa(TreeNode root,int cur){
        if (root == null){
            return;
        }
        depth = Math.max(depth,cur+1);
        aaaaaaa(root.left,cur+1);
        aaaaaaa(root.right,cur+1);
    }



    public boolean isSymmetric(TreeNode root) {
        if (root == null){
            return false;
        }
        return checkRoot(root.left,root.right);

    }
    private boolean checkRoot(TreeNode left,TreeNode right){
        if (left== null || right == null){
            return left== null && right == null;
        }
        if (left.val != right.val){
            return false;
        }
        return checkRoot(left.left,right.right) && checkRoot(left.right,right.left);

    }



    boolean flag = true;
    public boolean isValidBST(TreeNode root) {

        check(root);
        return flag;

    }

    private void check(TreeNode root){
        if (root == null){
            return;
        }
        if (root.left != null){
            if (root.left.val >= root.val){
                flag =  false;
            }
            check(root.left);
        }

        if (root.right != null){
            if (root.right.val <= root.val){
                flag =  false;
            }
            check(root.right);
        }
    }


    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] res = new int[n];
        for (int i= 0;i<n;i++){
            int cur = 0;
            for (int j = i;j<n;j++){
                if (temperatures[j] > temperatures[i]){
                    cur = j-i;
                    break;
                }
            }
            res[i] = cur;
        }
        return res;

    }

    public static int countSubstrings(String s) {
        int n = s.length();
        if (n == 0){
            return 0;
        }
        int count = 0;
        boolean [][] dp = new boolean[n][n];
        for (int i = 0;i< n;i++){
            dp[i][i] = true;
            count++;
        }
        if (n == 1){
            return 1;
        }
        for (int i = 0;i< n-1;i++){
            if (s.charAt(i) == s.charAt(i+1)){
                dp[i][i+1] = true;
                count++;
            }
        }
        for (int i = n-1;i>0;i--){
            for (int j = i;j<n;j++){
                if (i-1 >= 0 && j+1 <n && s.charAt(i-1) == s.charAt(j+1) && dp[i][j]){
                    dp[i-1][j+1] = true;
                    count++;
                }
            }

        }

        return count;
    }

    public static List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        int n1 = s.length();
        int n2 = p.length();
        if (n2 > n1){
            return res;
        }


        for (int i = 0;i<=n1-n2;i++){
            HashMap<Character,Integer> hashMap = new HashMap();
            for (int j = 0;j<n2;j++){
                if (hashMap.containsKey(p.charAt(j))){
                    hashMap.put(p.charAt(j),hashMap.get(p.charAt(j))  + 1);
                }else {
                    hashMap.put(p.charAt(j),1);

                }
            }
            if (check(s.substring(i,i+ n2),hashMap)){
              res.add(i);
          }
        }

        return res;
    }

    private static boolean check(String s1, HashMap<Character,Integer> hashMap){
        for (int i = 0;i<s1.length();i++){
            if (hashMap.containsKey(s1.charAt(i))){
                hashMap.put(s1.charAt(i),hashMap.get(s1.charAt(i))  - 1);
            }else {
                return false;
            }
        }
        return hashMap.values().stream().noneMatch(s->s>0);
    }



    public int[][] reconstructQueue(int[][] people) {

        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]){
                    return o1[1] - o2[1];
                }else {
                    return o2[0] - o1[0];
                }
            }
        });
        List<int[]> res = new ArrayList<>();

        for (int [] o: people){
            res.add(o[1],o);
        }
        return res.toArray(new int[res.size()][]);

    }

    public static String decodeString(String s) {
        StringBuilder res = new StringBuilder();
        int multi = 0;
        LinkedList<Integer> stack_multi = new LinkedList<>();
        LinkedList<String> stack_res = new LinkedList<>();
        for (Character c : s.toCharArray()){
            if (c == '['){
                stack_multi.addLast(multi);
                stack_res.addLast(res.toString());
                multi = 0;
                res = new StringBuilder();
            }else if (c == ']'){
                StringBuilder tmp = new StringBuilder();
                int crr_multi = stack_multi.removeLast();
                for (int i = 0;i<crr_multi;i++){
                    tmp.append(res);

                }
                res = new StringBuilder(stack_res.removeLast() + tmp);
            }else if (c >= '0' && c<='9'){
                multi = multi * 10 + Integer.parseInt(c + "");
            }else {
                res.append(c);
            }
        }
        return res.toString();
    }

    public static int numRollsToTarget(int d, int f, int target) {
          int MOD = 1000000007;

        int[][] dp =new int[d][target+1];
        for (int j = 1; j<=f;j++){
            if (target >=j){
                dp[0][j] = 1;
            }

        }
        for (int  i = 1; i< d;i++){
            for (int x = 1;x<=target;x++){
                for (int j = 1; j<=f;j++){
                    if (x > j){
                        dp[i][x] = (dp[i][x] +dp[i-1][x-j]) % MOD;
                    }
                }
            }

        }
        return dp[d-1][target];

    }

    // public int change(int amount, int[] coins) {
    //     if (amount == 0){
    //         return 1;
    //     }
    //     int n = coins.length;
    //     if (n == 0){
    //         return 0;
    //     }
    //     int [] dp = new int[amount+1];
    //     dp[0] = 1;
    //     for (int coin : coins){
    //         for (int j = 1;j<=amount;j++){
    //             if (j >= coin){
    //                 dp[j] += dp[j-coin];
    //             }
    //
    //
    //         }
    //     }
    //     return dp[amount];
    // }

    static int res = Integer.MAX_VALUE;
    public static int lastStoneWeightII(int[] stones) {
        int n = stones.length;
        if (n == 0){
            return 0;
        }
        boolean used [] = new boolean[n];
        used[0] = true;
        back(0,stones,used,stones[0]);
        return res;
    }

    private static void back(int usedNum,int[] stones,boolean used [],int cur){
        for (int i = 1;i<stones.length;i++){
            if (usedNum == stones.length){
                res = Math.min(res,cur);
                return;
            }
            used[i] = true;
            usedNum ++ ;
            back(usedNum,stones,used,cur > stones[i] ? cur-stones[i]:stones[i]-cur);
            used[i] = false;
            usedNum --;
            back(usedNum,stones,used,cur);
        }

    }

    public static int change(int amount, int[] coins) {
        int n = coins.length;
        if (n == 0){
            return 0;
        }
        int [] dp = new int[amount +1];
        dp[0] = 1;
        for (int coin : coins){
            for (int i = coin;i<= amount;i++){
                dp[i]+=  dp[i-coin];
            }
        }
        return dp[amount];
    }

    public static int combinationSum4(int[] nums, int target) {
        int n = nums.length;
        if (n == 0){
            return 0;
        }
        Arrays.sort(nums);
        int [] dp = new int[target +1];
        dp[0] = 0;
        for (int i = 0;i<=target;i++){
            for (int j = 0;j<n;j++){
                if (i >= nums[j]){
                    if (i == nums[j]){
                        dp[i]++;
                    }else {
                        dp[i] += dp[i - nums[j]];
                    }

                }
            }
        }
        return dp[target];
    }





    private void back3(int[] nums, int target,int total){
        if (total== target){
            res++;
        }else if (total > target){
            return;
        }else {
            for (int x : nums){
                back3(nums,target,total + x);
            }
        }
    }

    public int findTargetSumWays(int[] nums, int target) {
        int n = nums.length;
        if (n == 0){
            return 0;
        }
        back2(nums,target,0,0);
        return res;
    }

    private void back2(int[] nums, int target,int begin,int total){
        if (begin == nums.length && total== target){
            res++;
        }
        total += nums[0];
        back2(nums,target,begin + 1,total);
        total -= nums[0];
        back2(nums,target,begin + 1,total);
    }



    public boolean canPartition(int[] nums) {
        int n = nums.length;
        if (n <2){
            return false;
        }
        int sum = 0;
        int max = 0;
        for (int x: nums){
            sum+=x;
            max = Math.max(x,max);
        }
        if (sum % 2 != 0){
            return false;
        }
        int target = sum/2;
        if (max> target){
            return false;
        }
        boolean[][] dp =  new boolean[n][target+1];

        for (int i = 0 ;i < n;i++){
            dp[i][0] = true;
        }
        dp[0][nums[0]] = true;
        for (int i = 1 ;i < n;i++){
            int num = nums[i];
            for (int j = 1 ;j<=target;j++){
                if (j>=num){
                    dp[i][j] = dp[i-1][j] || dp[i-1][j -num];
                }else {
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[n-1][target];

    }


    public static int coinChange(int[] coins, int amount) {
        int max = amount + 1;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, max);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (coins[j] <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }



    // public static int coinChange(int[] coins, int amount) {
    //     int n = coins.length;
    //     if (n ==0 || amount == 0){
    //         return 0;
    //     }
    //     int [] dp = new int[amount+1];
    //     Arrays.fill(dp, amount+1);//必须将所有的dp赋最大值，因为要找最小值
    //
    //     for (int i = 0; i<=amount;i++){
    //         for (int j = 0; j < coins.length; j++) {
    //             if (coins[j] <= i) {
    //                 dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
    //             }
    //         }
    //     }
    //     return dp[amount] == 0? -1:dp[amount];
    //
    // }

    public int maxProfit(int[] prices) {
        int n = prices.length;
        if (n == 0) {
            return 0;
        }
        int [][] dp = new int[n][3];
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        dp[0][2] = 0;
        for (int  i = 1;i<n;i++){
            dp[i][0] = Math.max(dp[i-1][0],dp[i-1][2] - prices[i]);
            dp[i][1] = dp[i-1][0] + prices[i];
            dp[i][2] = Math.max(dp[i-1][1],dp[i-1][2]);
        }
        return Math.max(dp[n-1][1],dp[n-1][2]);

    }



    public int lengthOfLIS(int[] nums) {

        int n = nums.length;
        if ( n ==0){
            return 0;
        }
        int res = 1;
        int [] dp = new int[n];
        dp[0]=1;
        for(int i =1;i< n;i++){
            dp[i] =1 ;
            for (int j = 0;j<i;j++){
                if (nums[i]> nums[j]){
                    dp[i]= Math.max(dp[i],dp[j] + 1);
                }
            }
            res = Math.max(res,dp[i]);
        }
        return res;


    }

    private static boolean ssss(int[][] matrix, int target,int start,boolean hengshu){
        int lo = start;
        int hi = hengshu?matrix[0].length-1:matrix.length-1;
        while(lo <=hi){
            int mid = (lo+hi)/2;
            if (hengshu){
                if (matrix[start][mid] < target){
                    lo = mid+1;
                }else if(matrix[start][mid] > target){
                    hi = mid-1;
                }else {
                    return true;
                }
            }else {
                if (matrix[mid][start] < target){
                    lo = mid+1;
                }else if(matrix[mid][start] > target){
                    hi = mid-1;
                }else {
                    return true;
                }
            }
        }
        return false;
    }

    public static boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0){
            return false;
        }
        int shortNum = Math.min(matrix.length,matrix[0].length);
        for (int i = 0;i< shortNum;i++){
            boolean heng = ssss(matrix,target,i,true);
            boolean shu = ssss(matrix,target,i,false);
            if (heng || shu){
                return true;
            }

        }
        return false;
    }


    // public static boolean searchMatrix(int[][] matrix, int target) {
    //
    //     int n1 = matrix.length;
    //     int n2 = matrix[0].length;
    //     int i = 0;
    //     int j = 0;
    //     int mid1 = n1/2;
    //     int mid2 = n2/2;
    //     while (mid1 > 0&& mid2>0){
    //         if (matrix[mid1][mid2] > target){
    //             if ( mid1== mid1/2 ||  mid2== mid2/2){
    //                return matrix[mid1-1][mid2] == target || matrix[mid1][mid2-1] == target;
    //
    //             }else {
    //                 mid1= mid1/2;
    //                 mid2= mid2/2;
    //             }
    //
    //         }else if (matrix[mid1][mid2] < target){
    //             if ( mid1==  (mid1 + n1)/2 ||  mid2== (mid2+n2)/2){
    //                 return matrix[mid1-1][mid2] == target || matrix[mid1][mid2-1] == target;
    //             }else {
    //                 mid1= (mid1 + n1)/2;
    //                 mid2= (mid2+n2)/2;
    //             }
    //
    //         }else {
    //             return true;
    //         }
    //     }
    //
    //     return false;
    //
    //
    //
    // }

    public static int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>((pair1, pair2) -> pair1[0] != pair2[0] ? pair2[0] - pair1[0] : pair2[1] - pair1[1]);
        for (int i = 0; i < k; ++i) {
            pq.offer(new int[]{nums[i], i});
        }
        int[] ans = new int[n - k + 1];
        ans[0] = pq.peek()[0];
        for (int i = k; i < n; ++i) {
            pq.offer(new int[]{nums[i], i});
            while (pq.peek()[1] <= i - k) {
                pq.poll();
            }
            ans[i - k + 1] = pq.peek()[0];
        }
        return ans;
    }



    public int [] findKthLargest(int[] nums, int k) {
        int n = nums.length;
        int [] left = new int[n];
        int [] right = new int[n];
        for (int i  =0;i<n;i++){
            if (i==0 || i%k ==0){
                left[i]= nums[i];
            }else {
                left[i] = Math.max(left[i-1],nums[i]);
            }
        }
        for (int j = n-1;j>=0;j--){
            if (j == n-1 || (j+1)%k == 0){
                right[j] = nums[j];
            }else {
                right[j]  = Math.max(right[j+1],nums[j]);
            }

        }

        int [] res = new int[n-k+1];
        for (int h =0;h<n-k;h++){
            res[h] = Math.max(right[h],left[h+k-1]);
        }
        return res;

    }




    public static boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> edges = new ArrayList<List<Integer>>();
        for (int i = 0; i < numCourses; ++i) {
            edges.add(new ArrayList<Integer>());
        }
        int[] indeg = new int[numCourses];
        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
            ++indeg[info[0]];
        }

        Queue<Integer> queue = new LinkedList<Integer>();
        for (int i = 0; i < numCourses; ++i) {
            if (indeg[i] == 0) {
                queue.offer(i);
            }
        }

        int visited = 0;
        while (!queue.isEmpty()) {
            ++visited;
            int u = queue.poll();
            for (int v: edges.get(u)) {
                --indeg[v];
                if (indeg[v] == 0) {
                    queue.offer(v);
                }
            }
        }

        return visited == numCourses;
    }

    //
    // public static  boolean canFinish(int numCourses, int[][] prerequisites) {
    //     List<List<Integer>> adjacency = new ArrayList<>();
    //     for(int i = 0; i < numCourses; i++)
    //         adjacency.add(new ArrayList<>());
    //     int[] flags = new int[numCourses];
    //     for(int[] cp : prerequisites)
    //         adjacency.get(cp[1]).add(cp[0]);
    //     for(int i = 0; i < numCourses; i++)
    //         if(!dfs(adjacency, flags, i)) return false;
    //     return true;
    // }
    // private static boolean dfs(List<List<Integer>> adjacency, int[] flags, int i) {
    //     if(flags[i] == 1) return false;
    //     if(flags[i] == -1) return true;
    //     flags[i] = 1;
    //     for(Integer j : adjacency.get(i))
    //         if(!dfs(adjacency, flags, j)) return false;
    //     flags[i] = -1;
    //     return true;
    // }





    public ListNode reverseList(ListNode head) {
        ListNode prew = null;
        ListNode cur = head;


        while(cur != null){
            ListNode next =  cur.next;
            cur.next = prew;
            prew = cur;
            cur = next;

        }
        return prew;

    }
    // // bfs
    // public  static int numIslands(char[][] grid) {
    //     if (grid == null || grid.length == 0){
    //         return 0;
    //     }
    //     int n1 = grid.length;
    //     int n2 = grid[0].length;
    //     int res = 0;
    //     for (int i = 0;i< n1;i++){
    //         for (int j =0;j< n2;j++){
    //             if (grid[i][j] ==  '1'){
    //                 ++res;
    //                 bfs(grid,i,j);
    //             }
    //         }
    //
    //     }
    //     return res;
    // }
    //
    // private void bfs(char[][] grid, int i, int j){
    //     Queue<int []> list = new LinkedList<>();
    //     list.add(new int []{i,j});
    //     while (!list.isEmpty()){
    //         int [] cur = list.remove();
    //         i = cur[0];j = cur[1];
    //         if (i>=0 && i<grid.length && j>= 0 && j < grid[0].length){
    //             grid[i][j] = '0';
    //             list.add(new int[] { i + 1, j });
    //             list.add(new int[] { i - 1, j });
    //             list.add(new int[] { i, j + 1 });
    //             list.add(new int[] { i, j - 1 });
    //
    //         }
    //     }
    // }




        // // dfs
    // public static int numIslands(char[][] grid) {
    //     if (grid == null || grid.length == 0){
    //         return 0;
    //     }
    //     int n1 = grid.length;
    //     int n2 = grid[0].length;
    //     int res = 0;
    //     for (int i = 0;i< n1;i++){
    //         for (int j =0;j< n2;j++){
    //             if (grid[i][j] ==  '1'){
    //                 ++res;
    //                 dfs(grid,i,j);
    //             }
    //         }
    //
    //     }
    //     return res;
    // }
    //
    // public static void dfs(char[][] grid,int i,int j){
    //     int n1 = grid.length;
    //     int n2 = grid[0].length;
    //     if (i< 0||j<0||i>=n1||j>=n2|| grid[i][j] == '0'){
    //         return;
    //     }
    //
    //     grid[i][j] = '0';
    //     dfs(grid,i-1,j);
    //     dfs(grid,i,j-1);
    //     dfs(grid,i+1,j);
    //     dfs(grid,i,j+1);
    // }

    public static int maxProduct(int[] nums) {
        int max = nums[0];
        int n = nums.length;
        for(int i= 0 ; i<n;i++){
            int current = nums[i];
            if (current > max){
                max = current;
            }
            for (int j = i+1;j<n;j++){
                current = current*nums[j];
                if (current > max){
                    max = current;
                }
            }
        }
        return max;
    }




    public static boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        boolean [] flag = new boolean[n+1];
        for (int i = 0;i <= n;i++){
            if (wordDict.contains(s.substring(0,i))){
                flag[i] = true;
            }
           for(int j = 0;j<=i;j++){
               if (flag[j] && wordDict.contains(s.substring(j,i))){
                   flag[i] = true;
                   break;
               }
           }
        }
        return flag[n];

    }


    public boolean exist(char[][] board, String word) {
        int h = board.length;
        int w = board[0].length;
        boolean[][] flagArr = new boolean[h][w];
        for(int i =0;i<h;i++){
            for (int j = 0;j<w;j++){
                boolean f  = check(i,j,0,word,flagArr,board);
                if (f){
                    return true;
                }

            }
        }
        return false;
    }

    public boolean check(int i,int j,int k,String word,boolean[][] flagArr,char[][] board){

        if (word.charAt(k)!= board[i][j]){
            return false;
        }else if (k == word.length()-1){
            return true;
        }

        flagArr[i][j] = true;
        int [][]dir ={{-1,0},{0,-1},{1,0},{0,1}};
        for (int [] d :dir){
            int newi = i + d[0];
            int newj = j+d[1];
            if (newi >= 0 && newi<board.length && newj>= 0 && newj < board[0].length){
                if (!flagArr[newi][newj]){
                    boolean f  = check(newi,newj,k+1,word,flagArr,board);
                    if (f){
                        return true;
                    }
                }


            }
        }
        flagArr[i][j] = false;
        return false;

    }


    public  int uniquePaths(int m, int n) {
        if (n ==0 ){
            return 0;
        }
        if (m == 1 || n == 1){
            return 1;
        }

        int [][] nums = new int[m][n];
        return getCount(m,n,1,1,nums);

    }

    private int  getCount(int m, int n,int i,int j,int [][] nums){
        if (i == m || j == n){
            return 1;
        }
        nums[m][n-1] = 1;
        nums[m-1][n] = 1;
        nums[i][j] = getCount(m,n,i+1,j,nums) + getCount(m,n,i,j+1,nums);
       return nums[i][j];
    }

    private static void addCureent(int [][] nums){
        int i =1,j=1;
        nums[i][j] = nums[i][j+1] + nums[i+1][j];
    }



        private static void addCureent(int m, int n,int i, int j, List<Integer> res){
        if (i > m || j >n){
            return;
        }
        if (i == m && j == n){
            res.add(1);
        }
        if (i == m){
            addCureent(m,n,i+1,j,res);
            return;
        }
        if (j == n){
            addCureent(m,n,i+1,j,res);
            return;

        }
        addCureent(m,n,i+1,j,res);
        addCureent(m,n,i,j+1,res);

    }


    public static boolean canJump(int[] nums) {
        int max = 0;
        int n = nums.length;
        if (n ==1)
        {
            return true;
        }
        for (int i =0;i<n-1;i++){
            // if (i <= max){
            max = Math.max(max,i+nums[i]);
            if (max > n-1){
                return true;
            }
            // }

        }
        return false;
    }
    //
    // public static boolean canJump(int[] nums) {
    //
    //     int n = nums.length;
    //     if (n==0){
    //         return false;
    //     }
    //
    //     List<Integer> max = new ArrayList<>();
    //     back(0,nums,n,max);
    //     return max.size() > 0;
    //
    // }
    //
    // private static void back(int i,int[] nums,int n, List<Integer> max){
    //     int j = nums[i];
    //     if (i + j >= n -1){
    //         max.add(1);
    //         return;
    //     }
    //     if (j == 0){
    //         return;
    //     }
    //     for(int k = 1;k<=j;k++){
    //         back(i + k,nums,n,max);
    //     }
    // }




    public int maxSubArray(int[] nums) {
        int pre = 0; int max = nums[0];
        for(int i : nums){
            pre = Math.max(i,pre+i);
            max = Math.max(max,pre);
        }
        return max;
    }

    public static List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;
        if (n ==0){
            return res;
        }
        List<Integer> path = new ArrayList<>();
        boolean [] used = new boolean[n];
        dfs(res,path,0,nums,n,used);
        return res;
    }

    private static void dfs(List<List<Integer>> res,List<Integer> path,int depth,int[] nums,int n,boolean [] used){
        if (depth == n){
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0;i<n;i++){
            if (!used[i]){
                path.add(nums[i]);
                used[i] = true;
                dfs(res,path,depth+1,nums,n,used);

                // 回到上一层
                used[i] = false;
                path.remove(depth);
            }
        }
    }

    public static int saaaaaearch(int[] nums, int target) {
        int length = nums.length;
        if(length ==0){
            return -1;
        }
        if (length == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int left = 0,right = length-1;
        while (left <= right){
            int mid = (left + right)/2;
            if (nums[mid] == target){
                return mid;
            }

            if (nums[0] <= nums[mid]){
                // 左边有序
                if (nums[0] <=target && target <nums[mid]){
                    // 在分界后的左边
                    right = mid -1;
                }else {
                    // 在分界后的右边
                    left = mid +1;
                }
            }else {
                // 右边有序
                if (nums[mid] <target && target <=nums[length-1]){
                    // 在分界后的右边
                    left = mid +1;
                }else {
                    // 在分界后的左边
                    right = mid -1;
                }
            }
        }
        return -1;


    }

    public static int longestValidParentheses(String s) {
        int length = s.length();
        int max = 0;
        Deque<Integer> deque = new ArrayDeque<>();
        deque.push(-1);

        for (int i = 0;i<length;i++){
            if ('(' == s.charAt(i)){
                deque.push(i);
            }else {
                deque.pop();
                if (deque.isEmpty()){
                    deque.push(i);
                }else {
                    max = Math.max(max,i - deque.peek());
                }


            }
        }
        return max;
    }


    public static boolean isValid(String s) {
        int n = s.length();
        if (n % 2 == 1) {
            return false;
        }
        Map<Character, Character> pairs = new HashMap<Character, Character>() {{
            put(')', '(');
            put(']', '[');
            put('}', '{');
        }};
        Deque<Character> stack = new LinkedList<>();
        for(int i = 0;i<n;i++){
            char ch = s.charAt(i);
            if (pairs.containsKey(ch)){
                if (stack.isEmpty() || !stack.peek() .equals(pairs.get(ch))){
                    return false;
                }
                stack.pop();
            }  else {
                stack.push(ch);
            }

        }
        return stack.isEmpty();
    }


    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        int length = nums.length;
        Arrays.sort(nums);
        if (length < 3 || nums[0] >0){
            return res;
        }
        for (int i = 0;i < length-2;i++){
            int left = i+1,right = length-1;

            if (i>0 && nums[i] == nums[i-1]){
                continue;
            }
            while (left < right){
                if (nums[i] + nums[left] + nums[right] >0){
                    right--;
                }else if (nums[i] + nums[left] + nums[right] <0){
                    left++;
                }else {
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[i]);
                    list.add(nums[left]);
                    list.add(nums[right]);
                    res.add(list);
                    while (nums[right] == nums[right-1]){
                        right--;
                    }
                    while (nums[left] == nums[left+1]){
                        left++;
                    }
                    right--;
                    left++;
                }
            }


        }
        return res;

    }


    public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int n1 = nums1.length;
        int n2 = nums2.length;
        int[] nums3 = new int[n1+n2];
        int i =0,j=0,k=0;
        while (i < n1 && j< n2){
            if (nums1[i] < nums2[j]){
                nums3[k] = nums1[i];
                i++;
            }else {
                nums3[k] = nums2[j];
                j++;
            }
            k++;
        }
        while (i < n1){
            nums3[k] = nums1[i];
            k++;
            i++;
        }
        while (j<n2){
            nums3[k] = nums2[j];
            k++;
            j++;
        }
        double res ;
        if (nums3.length%2 == 0){
            res =(double) (nums3[nums3.length/2 -1] +  nums3[nums3.length/2])/2;
        }else {
            res = nums3[nums3.length/2];
        }

        return res;
    }


    private static BigDecimal change(BigDecimal a){
        a= a.add(BigDecimal.ONE);
        return a;
    }

    public int lengthOfLongestSubstring(String s) {

        Set<Character> subSet = new HashSet<>();
        int n = s.length();
        int right = 0,res =0;
        for (int i = 0;i<n;i++){
            if (i>0){
                // 左边指针右滑删除左边字符
                subSet.remove(s.charAt(i-1));
            }
            while (right < n && !subSet.contains(s.charAt(right))){
                subSet.add(s.charAt(right));
                right++;
            }
            res = Math.max(res,right-i);
        }
        return res;


    }


    public static String longestPalindrome(String s) {

        int len = s.length();
        if (len < 2){
            return s;
        }
        int begin =0,max =1;
        boolean [][] dp = new boolean[len][len];
        char[] arr = s.toCharArray();
        for(int i =0;i<len;i++){
            dp[i][i] = true;
        }
        for(int lef =2 ;lef<len;lef++){
            for (int i =0 ;i<len;i++){
                int right = lef +i-1;
                if (right>=len){
                    break;
                }
                if (arr[i] != arr[right]){
                    dp[i][right] = false;
                }else {
                   if (right - i <3){
                       dp[i][right] = true;
                   }else {
                       dp[i][right]= dp[i+1][right-1];
                   }
                }
                if (dp[i][right]  && right-i+1 >max){
                    max = right -i +1;
                    begin = i;
                }
            }

        }
        return s.substring(begin,begin+max);

    }

    // public List<String> letterCombinations(String digits) {
    //     List<String> combinations = new ArrayList<String>();
    //     if (digits.length() == 0) {
    //         return combinations;
    //     }
    //     Map<Character, String> phoneMap = new HashMap<Character, String>() {{
    //         put('2', "abc");
    //         put('3', "def");
    //         put('4', "ghi");
    //         put('5', "jkl");
    //         put('6', "mno");
    //         put('7', "pqrs");
    //         put('8', "tuv");
    //         put('9', "wxyz");
    //     }};
    //     backtrack(combinations, phoneMap, digits, 0, new StringBuffer());
    //     return combinations;
    // }
    //
    // public void backtrack(List<String> combinations, Map<Character, String> phoneMap, String digits, int index, StringBuffer combination) {
    //     if (index == digits.length()) {
    //         combinations.add(combination.toString());
    //     } else {
    //         char digit = digits.charAt(index);
    //         String letters = phoneMap.get(digit);
    //         int lettersCount = letters.length();
    //         for (int i = 0; i < lettersCount; i++) {
    //             combination.append(letters.charAt(i));
    //             backtrack(combinations, phoneMap, digits, index + 1, combination);
    //             combination.deleteCharAt(index);
    //         }
    //     }
    // }

    public List<String> letterCombinations(String digits) {
        //使用一个集合来存储所有的字母组合结果
        List<String> combinations = new ArrayList<>();
        if (digits == null || digits.length() == 0) {
            return combinations;
        }

        //将号码字母对应关系存储进Map
        HashMap<Character, String[]> map =new HashMap<>();
        map.put('2', new String[]{"a", "b", "c"});
        map.put('3', new String[]{"d", "e", "f"});
        map.put('4', new String[]{"g", "h", "i"});
        map.put('5', new String[]{"j", "k", "l"});
        map.put('6', new String[]{"m", "n", "o"});
        map.put('7', new String[]{"p", "q", "r", "s"});
        map.put('8', new String[]{"t", "u", "v"});
        map.put('9', new String[]{"w", "x", "y", "z"});


        //定义一个队列来存储所有的组合结果
        Queue<String> queue = new LinkedList<>();
        //遍历Digits，取到对应号码对应的字母数组
        // 345
        for (int i = 0; i < digits.length(); i++) {
            handQueue(queue, map.get(digits.charAt(i)));
        }
        //要求返回List
        combinations.addAll(queue);
        return combinations;
    }

    private void handQueue(Queue<String> queue,String[] arr){
        int length = queue.size();
        if (length == 0){
            for (String s : arr){
                queue.add(s);
            }
        }else {
            for (int i = 0;i<length;i++){
                String poll = queue.poll();
                for (String s : arr){
                    queue.add(poll +s);
                }
            }

        }


    }

    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length <= 1){
            return;
        }
        int i = nums.length -2;
      while(i>=0 && nums[i] >= nums[i+1]){i--;}
      if (i>=0){
          int j = nums.length -1;
         while (j >=0 && nums[i] >= nums[j]){
             j--;
         }
          swap(nums,i,j);

      }
        reveser(nums,i+1);


    }

    private void swap(int[] nums,int i,int j){
        int tem = nums[i];
        nums[i] = nums[j];
        nums[j] =tem;
    }

    private void reveser(int[] nums,int start){
        int left = start;
        int right = nums.length-1;
        while (left < right){
            swap(nums,left,right);
            left++;
            right--;
        }
    }


    // public int longestValidParentheses(String s) {
    //     int max = 0;
    //     int length = s.length();
    //     int left = 0,right = 0;
    //     for (int i = 0;i<length;i++){
    //         if (s.charAt(i) == '('){
    //             left++;
    //         }else{
    //             right++;
    //         }
    //         if (right == left){
    //             max = Math.max(max,right*2);
    //         }else if (right > left){
    //             left =right=0;
    //         }
    //     }
    //
    //     for (int i = length -1 ;i >0;i--){
    //         if (s.charAt(i) == '('){
    //             left++;
    //         }else{
    //             right++;
    //         }
    //         if (right == left){
    //             max = Math.max(max,right*2);
    //         }else if (right < left){
    //             left =right=0;
    //         }
    //     }
    //
    //     return max;
    //
    // }


    //
    // public int longestValidParentheses(String s) {
    //     int maxans = 0;
    //     Deque<Integer> stack = new LinkedList<Integer>();
    //     stack.push(-1);
    //     for (int i = 0; i < s.length(); i++) {
    //         if (s.charAt(i) == '(') {
    //             stack.push(i);
    //         } else {
    //             stack.pop();
    //             if (stack.isEmpty()) {
    //                 stack.push(i);
    //             } else {
    //                 maxans = Math.max(maxans, i - stack.peek());
    //             }
    //         }
    //     }
    //     return maxans;
    // }


    public int search(int[] nums, int target) {
        int length = nums.length;
        if(length ==0){
            return -1;
        }
        int left = 0,right = length;
        while (left <= right){
            int mid = (left + right)/2;
            if (nums[mid] == target){
                return mid;
            }
            if (nums[left] < nums[mid]){
                // 左边有序
                if (nums[left] <target && target <nums[mid]){
                    // 在分界后的左边
                    right = mid -1;
                }else {
                    // 在分界后的右边
                    left = mid +1;
                }
            }else {
                // 右边有序
                if (nums[mid] <target && target <nums[length-1]){
                    // 在分界后的右边
                    left = mid +1;
                }else {
                    // 在分界后的左边
                    left = mid -1;
                }
            }
        }
        return -1;


    }
    //
    //
    // class Solution {
    //     public int search(int[] nums, int target) {
    //         int n = nums.length;
    //         if (n == 0) {
    //             return -1;
    //         }
    //         if (n == 1) {
    //             return nums[0] == target ? 0 : -1;
    //         }
    //         int l = 0, r = n - 1;
    //         while (l <= r) {
    //             int mid = (l + r) / 2;
    //             if (nums[mid] == target) {
    //                 return mid;
    //             }
    //             if (nums[0] <= nums[mid]) {
    //                 if (nums[0] <= target && target < nums[mid]) {
    //                     r = mid - 1;
    //                 } else {
    //                     l = mid + 1;
    //                 }
    //             } else {
    //                 if (nums[mid] < target && target <= nums[n - 1]) {
    //                     l = mid + 1;
    //                 } else {
    //                     r = mid - 1;
    //                 }
    //             }
    //         }
    //         return -1;
    //     }
    // }


    public List<List<Integer>> combinationSum(int[] candidates, int target) {

        List<List<Integer>> res = new ArrayList<>();
        int length = candidates.length;
        List<Integer> oneAnswer = new ArrayList<>();
        if (length == 0){
            return res;
        }
        back(candidates,target,oneAnswer,res,0);
        return res;

    }

    private void back(int[] candidates, int target,List<Integer> oneAnswer,List<List<Integer>> res,int index){
        if (index == candidates.length){
            return;
        }
        if (target == 0){
            res.add(new ArrayList<>(oneAnswer));
            return;
        }
        back(candidates,target,oneAnswer,res,index +1);
        if (target - candidates[index] >= 0){
            oneAnswer.add(candidates[index]);
            back(candidates,target,oneAnswer,res,index);
            oneAnswer.remove(oneAnswer.size() -1);
        }


    }


    class Solution {
        public List<List<Integer>> combinationSum(int[] candidates, int target) {
            List<List<Integer>> ans = new ArrayList<List<Integer>>();
            List<Integer> combine = new ArrayList<Integer>();
            dfs(candidates, target, ans, combine, 0);
            return ans;
        }

        public void dfs(int[] candidates, int target, List<List<Integer>> ans, List<Integer> combine, int idx) {
            if (idx == candidates.length) {
                return;
            }
            if (target == 0) {
                ans.add(new ArrayList<Integer>(combine));
                return;
            }
            // 直接跳过
            dfs(candidates, target, ans, combine, idx + 1);
            // 选择当前数
            if (target - candidates[idx] >= 0) {
                combine.add(candidates[idx]);
                dfs(candidates, target - candidates[idx], ans, combine, idx);
                combine.remove(combine.size() - 1);
            }
        }
    }

    public  boolean can1Jump(int[] nums) {
        int max = 0;
        int n = nums.length;
        for (int i =0;i<n;i++){
            // if (i <= max){
                max = Math.max(max,i+nums[i]);
                if (max > n-1){
                    return true;
                }
            // }

    }
    return false;
    }





}
