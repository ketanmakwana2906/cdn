package main

import (
    "fmt"
    "errors"
)

// 1. Basic Function
func greet() {
    fmt.Println("Hello, Ketan!")
}

// 2. Function with Parameters
func greetWithName(name string) {
    fmt.Println("Hello,", name)
}

// 3. Function with Multiple Parameters
func add(a, b int) {
    fmt.Println("Sum:", a + b)
}

// 4. Function with Return Value
func getGreeting() string {
    return "Hello from return"
}

// 5. Function with Multiple Return Values
func divide(a, b int) (int, bool) {
    if b == 0 {
        return 0, false
    }
    return a / b, true
}

// 6. Named Return Values
func fullName() (first string, last string) {
    first = "Ketan"
    last = "Makwana"
    return
}

// 7. Variadic Function
func sum(nums ...int) int {
    total := 0
    for _, num := range nums {
        total += num
    }
    return total
}

// 8. Function Returning Another Function
func multiplier(factor int) func(int) int {
    return func(x int) int {
        return x * factor
    }
}

// 9. Anonymous Function
func runAnonymous() {
    func() {
        fmt.Println("Anonymous function ran")
    }()
}

// 10. Function as a Value
func functionVariable() {
    f := func(name string) {
        fmt.Println("Hi,", name)
    }
    f("Ketan")
}

// 11. Method with Receiver
type User struct {
    name string
}

func (u User) greet() {
    fmt.Println("Hello,", u.name)
}

// 12. Pointer Receiver
func (u *User) updateName(newName string) {
    u.name = newName
}

// 13. Defer in Function
func demoDefer() {
    defer fmt.Println("This runs last")
    fmt.Println("This runs first")
}

// 14. Function with Error Return
func riskyDivision(a, b int) (int, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

// FOR LOOP EXAMPLES

func loopExamples() {
    fmt.Println("\n-- Standard for loop --")
    for i := 0; i < 5; i++ {
        fmt.Println(i)
    }

    fmt.Println("\n-- For as while loop --")
    i := 0
    for i < 5 {
        fmt.Println(i)
        i++
    }

    fmt.Println("\n-- Infinite loop with break --")
    j := 0
    for {
        fmt.Println("Looping:", j)
        j++
        if j > 2 {
            break
        }
    }

    fmt.Println("\n-- Loop over slice --")
    nums := []int{10, 20, 30}
    for index, value := range nums {
        fmt.Println("Index:", index, "Value:", value)
    }

    fmt.Println("\n-- Loop over map --")
    m := map[string]int{"a": 1, "b": 2}
    for key, value := range m {
        fmt.Println(key, "=>", value)
    }

    fmt.Println("\n-- Loop over string --")
    for i, ch := range "Ketan" {
        fmt.Printf("Index: %d, Char: %c\n", i, ch)
    }

    fmt.Println("\n-- Break and Continue --")
    for i := 0; i < 10; i++ {
        if i == 5 {
            continue
        }
        if i == 8 {
            break
        }
        fmt.Println(i)
    }

    fmt.Println("\n-- Nested loop --")
    for i := 1; i <= 3; i++ {
        for j := 1; j <= 3; j++ {
            fmt.Printf("%d x %d = %d\n", i, j, i*j)
        }
    }

    fmt.Println("\n-- Labeled loop --")
outer:
    for i := 0; i < 3; i++ {
        for j := 0; j < 3; j++ {
            if i == j {
                break outer
            }
        }
    }
}

func main() {
    greet()
    greetWithName("Ketan")
    add(3, 4)
    fmt.Println(getGreeting())

    result, ok := divide(10, 2)
    fmt.Println("Divide Result:", result, "Success:", ok)

    first, last := fullName()
    fmt.Println("Full Name:", first, last)

    fmt.Println("Sum of variadic:", sum(1, 2, 3, 4))

    double := multiplier(2)
    fmt.Println("Double 5:", double(5))

    runAnonymous()
    functionVariable()

    user := User{name: "Guest"}
    user.greet()
    user.updateName("Ketan")
    user.greet()

    demoDefer()

    res, err := riskyDivision(10, 0)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("Result:", res)
    }

    loopExamples()
}
