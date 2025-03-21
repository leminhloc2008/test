#include <iostream>
#include <vector>
#include <set>
using namespace std;

// Function to detect cycle in an undirected graph using DFS
bool hasCycleUndirected(const vector<vector<int>>& graph, int vertex, int parent, vector<bool>& visited) {
    visited[vertex] = true;
    
    for (int neighbor : graph[vertex]) {
        // If the neighbor is not visited, check if there's a cycle starting from this neighbor
        if (!visited[neighbor]) {
            if (hasCycleUndirected(graph, neighbor, vertex, visited))
                return true;
        }
        // If the neighbor is already visited and not the parent, there's a cycle
        else if (neighbor != parent) {
            return true;
        }
    }
    return false;
}

// Wrapper function for the undirected graph cycle detection
bool detectCycleUndirected(const vector<vector<int>>& graph, int numVertices) {
    vector<bool> visited(numVertices, false);
    
    // Check each component of the graph (in case the graph is not connected)
    for (int i = 0; i < numVertices; i++) {
        if (!visited[i]) {
            if (hasCycleUndirected(graph, i, -1, visited))
                return true;
        }
    }
    return false;
}

// Function to detect cycle in a directed graph using DFS
bool hasCycleDirected(const vector<vector<int>>& graph, int vertex, vector<bool>& visited, vector<bool>& recStack) {
    // Mark the current node as visited and add to recursion stack
    visited[vertex] = true;
    recStack[vertex] = true;
    
    // Check all neighbors
    for (int neighbor : graph[vertex]) {
        // If the neighbor is not visited, check recursively
        if (!visited[neighbor]) {
            if (hasCycleDirected(graph, neighbor, visited, recStack))
                return true;
        }
        // If the neighbor is in the recursion stack, there's a cycle
        else if (recStack[neighbor]) {
            return true;
        }
    }
    
    // Remove the vertex from recursion stack before returning
    recStack[vertex] = false;
    return false;
}

// Wrapper function for the directed graph cycle detection
bool detectCycleDirected(const vector<vector<int>>& graph, int numVertices) {
    vector<bool> visited(numVertices, false);
    vector<bool> recStack(numVertices, false);
    
    // Check for cycle in each component
    for (int i = 0; i < numVertices; i++) {
        if (!visited[i]) {
            if (hasCycleDirected(graph, i, visited, recStack))
                return true;
        }
    }
    return false;
}

// Example usage
int main() {
    int numVertices, numEdges;
    bool isDirected;
    
    cout << "Enter number of vertices: ";
    cin >> numVertices;
    
    cout << "Enter number of edges: ";
    cin >> numEdges;
    
    cout << "Is the graph directed? (1 for yes, 0 for no): ";
    cin >> isDirected;
    
    // Initialize the adjacency list
    vector<vector<int>> graph(numVertices);
    
    cout << "Enter the edges (vertex pairs, 0-indexed):" << endl;
    for (int i = 0; i < numEdges; i++) {
        int u, v;
        cin >> u >> v;
        
        graph[u].push_back(v);
        
        // For undirected graph, add the edge in both directions
        if (!isDirected) {
            graph[v].push_back(u);
        }
    }
    
    // Detect cycle based on the graph type
    bool hasCycle;
    if (isDirected) {
        hasCycle = detectCycleDirected(graph, numVertices);
    } else {
        hasCycle = detectCycleUndirected(graph, numVertices);
    }
    
    if (hasCycle) {
        cout << "The graph contains a cycle." << endl;
    } else {
        cout << "The graph does not contain a cycle." << endl;
    }
    
    return 0;
}
