from abc import ABC, abstractmethod

import numpy as np


ROLL_NUMBERS = ["24F-5529", "24F-5521"]
DATA_FILE = "portfolio_data.txt"


class PortfolioError(Exception):
    pass


class Investment(ABC):
    """Abstract base class that defines the blueprint for every investment type."""

    def __init__(self, name, principal, annual_return, risk_score):
        """Initialize the investment with encapsulated attributes and validation."""
        if principal < 0:
            raise PortfolioError("Investment amount cannot be negative.")
        self.__name = name
        self.__principal = principal
        self.__annual_return = annual_return
        self.__risk_score = risk_score

    @property
    def name(self):
        """Return the name of the investment."""
        return self.__name

    @property
    def principal(self):
        """Return the invested principal amount."""
        return self.__principal

    @property
    def annual_return(self):
        """Return the expected annual return percentage."""
        return self.__annual_return

    @property
    def risk_score(self):
        """Return the risk score for the investment."""
        return self.__risk_score

    def __add__(self, other):
        """Enable operator overloading to combine two investments logically."""
        if not isinstance(other, Investment):
            raise PortfolioError("Can only add Investment instances together.")
        return CompositeInvestment([self, other])

    def to_record(self):
        """Return a pipe-separated line so we can save it in a text file."""
        data = [
            self.__class__.__name__,
            self.__name,
            str(self.__principal),
            str(self.__annual_return),
            str(self.__risk_score),
        ]
        return "|".join(data)

    @staticmethod
    def from_record(record):
        """Rebuild an investment object from the text file line."""
        parts = record.split("|")
        if len(parts) != 5:
            raise PortfolioError("Corrupted investment record.")

        investment_type, name, principal, annual_return, risk_score = parts
        principal = float(principal)
        annual_return = float(annual_return)
        risk_score = float(risk_score)

        if investment_type == "StockInvestment":
            return StockInvestment(name, principal, annual_return, risk_score)
        if investment_type == "MutualFundInvestment":
            return MutualFundInvestment(name, principal, annual_return, risk_score)
        if investment_type == "CryptoInvestment":
            return CryptoInvestment(name, principal, annual_return, risk_score)
        if investment_type == "RealEstateInvestment":
            return RealEstateInvestment(name, principal, annual_return, risk_score)

        raise PortfolioError(f"Unknown investment type: {investment_type}")

    @abstractmethod
    def calculate_return(self):
        """Calculate the projected annual monetary return for this investment."""

    @abstractmethod
    def calculate_risk(self):
        """Calculate the risk-adjusted score for this investment."""


class StockInvestment(Investment):
    """Concrete investment type representing a stock holding."""

    def __init__(self, name, principal, annual_return, risk_score, volatility=0.2):
        """Initialize a stock with volatility that influences risk."""
        super().__init__(name, principal, annual_return, risk_score)
        self.__volatility = volatility

    def calculate_return(self):
        """Return the projected gain for a stock after accounting for volatility."""
        multiplier = max(0.5, 1 - self.__volatility)
        return self.principal * (self.annual_return * multiplier)

    def calculate_risk(self):
        """Return the risk metric for the stock."""
        return self.risk_score + (self.__volatility * 10)


class MutualFundInvestment(Investment):
    """Concrete investment representing a mutual fund product."""

    def __init__(self, name, principal, annual_return, risk_score):
        """Initialize a mutual fund investment."""
        super().__init__(name, principal, annual_return, risk_score)

    def calculate_return(self):
        """Return the projected gain for a mutual fund."""
        diversification_bonus = 1.05
        return self.principal * self.annual_return * diversification_bonus

    def calculate_risk(self):
        """Return the risk metric for the mutual fund."""
        return max(1.0, self.risk_score - 2)


class CryptoInvestment(Investment):
    """Concrete investment representing a cryptocurrency holding."""

    def __init__(self, name, principal, annual_return, risk_score, volatility_multiplier=2.5):
        """Initialize a cryptocurrency investment with a volatility multiplier."""
        super().__init__(name, principal, annual_return, risk_score)
        self.__volatility_multiplier = volatility_multiplier

    def calculate_return(self):
        """Return the projected gain for a cryptocurrency investment."""
        return self.principal * self.annual_return * self.__volatility_multiplier

    def calculate_risk(self):
        """Return the risk metric for the cryptocurrency."""
        return self.risk_score * self.__volatility_multiplier


class RealEstateInvestment(Investment):
    """Concrete investment representing a real estate asset."""

    def __init__(self, name, principal, annual_return, risk_score):
        """Initialize a real estate investment."""
        super().__init__(name, principal, annual_return, risk_score)

    def calculate_return(self):
        """Return the projected gain for a real estate investment."""
        appreciation_factor = 1.1
        rental_income = 0.04 * self.principal
        return (self.principal * self.annual_return * appreciation_factor) + rental_income

    def calculate_risk(self):
        """Return the risk metric for the real estate asset."""
        return max(0.5, self.risk_score * 0.8)


class CompositeInvestment(Investment):
    """Composite investment produced via operator overloading."""

    def __init__(self, investments):    
        """Initialize a composite by aggregating multiple investments."""
        combined_name = " & ".join(inv.name for inv in investments)
        total_principal = sum(inv.principal for inv in investments)
        average_return = np.mean([inv.annual_return for inv in investments])
        average_risk = np.mean([inv.risk_score for inv in investments])
        super().__init__(combined_name, total_principal, average_return, average_risk)
        self.__investments = investments

    def calculate_return(self):
        """Return the combined projected return of all nested investments."""
        return sum(inv.calculate_return() for inv in self.__investments)

    def calculate_risk(self):
        """Return the average risk of the composite investment."""
        return np.mean([inv.calculate_risk() for inv in self.__investments])


class User:
    """Represents a SIPMS user with secure, encapsulated data."""

    def __init__(self, username, email):
        """Initialize a user with identifying information."""
        self.__username = username
        self.__email = email

    @property
    def username(self):
        """Return the user's username."""
        return self.__username

    @property
    def email(self):
        """Return the user's email."""
        return self.__email


class Portfolio:
    """Represents a collection of investments mapped to a user."""

    def __init__(self, owner, portfolio_name):
        """Initialize the portfolio with basic metadata."""
        self.__owner = owner
        self.__portfolio_name = portfolio_name
        self.__investments = []

    @property
    def owner(self):
        """Return the owner of the portfolio."""
        return self.__owner

    @property
    def portfolio_name(self):
        """Return the name of the portfolio."""
        return self.__portfolio_name

    @property
    def investments(self):
        """Return a copy of the investment list to preserve encapsulation."""
        return self.__investments[:]

    def add_investment(self, investment):
        """Add a new investment to the portfolio."""
        if not isinstance(investment, Investment):
            raise PortfolioError("Invalid investment instance.")
        self.__investments.append(investment)

    def remove_investment(self, name):
        """Remove an investment by name and return operation status."""
        for investment in self.__investments:
            if investment.name == name:
                self.__investments.remove(investment)
                return True
        return False

    def total_principal(self):
        """Return the total principal invested across the portfolio."""
        return sum(investment.principal for investment in self.__investments)

    def projected_annual_return(self):
        """Return the sum of projected returns for all investments."""
        return sum(investment.calculate_return() for investment in self.__investments)

    def risk_profile(self):
        """Return the average risk score using NumPy for accuracy."""
        if not self.__investments:
            return 0.0
        risk_values = [investment.calculate_risk() for investment in self.__investments]
        return float(np.mean(risk_values))

    def numpy_statistics(self):
        """Return NumPy-based analytics such as average return and risk."""
        if not self.__investments:
            return {"averageReturn": 0.0, "riskStdDev": 0.0}
        returns = [investment.annual_return for investment in self.__investments]
        return {
            "averageReturn": float(np.mean(returns)),
            "riskStdDev": float(np.std(returns)),
        }

    def best_investment(self):
        """Return the investment with the highest annual return using recursion."""
        def explore(index, best):
            if index == len(self.__investments):
                return best
            current = self.__investments[index]
            if best is None or current.annual_return > best.annual_return:
                best = current
            return explore(index + 1, best)

        return explore(0, None)

    def __add__(self, other):
        """Enable portfolio merging through operator overloading."""
        if not isinstance(other, Portfolio):
            raise PortfolioError("Can only merge another Portfolio.")
        merged = Portfolio(self.__owner, f"{self.__portfolio_name} + {other.__portfolio_name}")
        for investment in self.__investments + other.__investments:
            merged.add_investment(investment)
        return merged


class PortfolioStorage:
    """File handling utility to persist and retrieve portfolio data."""

    @staticmethod
    def save(portfolio, file_path=DATA_FILE):
        """Write the portfolio to a simple text file."""
        with open(file_path, "w") as file:
            file.write(",".join(ROLL_NUMBERS) + "\n")
            file.write(f"{portfolio.owner.username},{portfolio.owner.email}\n")
            file.write(portfolio.portfolio_name + "\n")
            for investment in portfolio.investments:
                file.write(investment.to_record() + "\n")

    @staticmethod
    def load(file_path=DATA_FILE):
        """Load the portfolio from the text file."""
        try:
            with open(file_path, "r") as file:
                raw_lines = file.readlines()
        except FileNotFoundError:
            raise PortfolioError("Portfolio file does not exist.")

        lines = []
        for line in raw_lines:
            clean_line = line.strip()
            if clean_line:
                lines.append(clean_line)

        if len(lines) < 3:
            raise PortfolioError("Portfolio file is incomplete.")

        user_info = lines[1].split(",")
        if len(user_info) != 2:
            raise PortfolioError("User information is corrupted.")

        user = User(user_info[0], user_info[1])
        portfolio = Portfolio(user, lines[2])

        for record in lines[3:]:
            portfolio.add_investment(Investment.from_record(record))

        return portfolio


class ReportGenerator:
    """Utility class that produces human-readable portfolio reports."""

    @staticmethod
    def generate_summary(portfolio):
        """Create a textual report summarizing key portfolio metrics."""
        best = portfolio.best_investment()
        best_line = f"Best Investment: {best.name} ({best.annual_return*100:.2f}% return)" if best else "No investments yet."
        stats = portfolio.numpy_statistics()
        report_lines = [
            f"Portfolio Name: {portfolio.portfolio_name}",
            f"Owner: {portfolio.owner.username} ({portfolio.owner.email})",
            f"Total Principal: {portfolio.total_principal():.2f}",
            f"Projected Annual Return: {portfolio.projected_annual_return():.2f}",
            f"Average Risk Score: {portfolio.risk_profile():.2f}",
            f"Average Return (NumPy): {stats['averageReturn']*100:.2f}%",
            f"Return Std Dev (NumPy): {stats['riskStdDev']*100:.2f}%",
            best_line,
            f"Investments Count: {len(portfolio.investments)}",
        ]
        return "\n".join(report_lines)


class PolymorphismShowcase:
    """Demonstrates multiple polymorphism techniques within SIPMS."""

    @staticmethod
    def runtime_polymorphism(investment):
        """Demonstrate method overriding by invoking calculate_return."""
        return investment.calculate_return()

    @staticmethod
    def ad_hoc_polymorphism(value_a, value_b=0.0, *extras):
        """Demonstrate ad-hoc polymorphism via optional arguments."""
        total = value_a + value_b + sum(extras)
        return total

    @staticmethod
    def operator_polymorphism(investment_a, investment_b):
        """Demonstrate operator overloading by combining investments."""
        return investment_a + investment_b

    @staticmethod
    def duck_typing_polymorphism(entity):
        """Demonstrate duck typing by accepting any object with name attr."""
        if hasattr(entity, "name"):
            return f"Entity identified as: {entity.name}"
        return "Entity identified as: Unknown Entity"


def show_menu():
    """Display interactive menu options for the user."""
    print("\n" + "-"*50)
    print("SMART INVESTMENT PORTFOLIO MANAGER")
    print("-"*50)
    print("1. Initialize Portfolio")
    print("2. Register Stock Investment")
    print("3. Register Mutual Fund Investment")
    print("4. Register Cryptocurrency Investment")
    print("5. Register Real Estate Investment")
    print("6. Display Portfolio Summary")
    print("7. Identify Top Performing Investment (Recursion)")
    print("8. Perform Statistical Analysis (NumPy)")
    print("9. Demonstrate Polymorphic Techniques")
    print("10. Persist Portfolio to File")
    print("11. Generate Comprehensive Report")
    print("12. Exit Program")
    print("-"*50)


def main():
    """Interactive menu-driven entry point for SIPMS."""
    portfolio = None
    user = None
    
    print("\n" + "-"*50)
    print("Welcome to Smart Investment Portfolio Manager!")
    print("-"*50)
    
    while True:
        show_menu()
        choice = input("\nSelect an option (1-12): ").strip()
        
        try:
            # Initialize Portfolio
            if choice == '1':
                username = input("\nEnter your username: ").strip()
                email = input("Enter your email: ").strip()
                portfolio_name = input("Enter portfolio name: ").strip()
                
                user = User(username, email)
                portfolio = Portfolio(user, portfolio_name)
                print(f"\nPortfolio '{portfolio_name}' successfully initialized for {username}!")
            
            # Register Stock Investment
            elif choice == '2':
                if portfolio is None:
                    print("\nPlease initialize a portfolio first (Option 1)!")
                    continue
                
                name = input("\nEnter stock name: ").strip()
                principal = float(input("Investment amount ($): "))
                annual_return = float(input("Expected annual return (e.g., 0.12 for 12%): "))
                risk_score = float(input("Risk score (0-10): "))
                
                stock = StockInvestment(name, principal, annual_return, risk_score)
                portfolio.add_investment(stock)
                print(f"\nStock '{name}' successfully added to portfolio!")
            
            # Register Mutual Fund Investment
            elif choice == '3':
                if portfolio is None:
                    print("\nPlease initialize a portfolio first (Option 1)!")
                    continue
                
                name = input("\nEnter mutual fund name: ").strip()
                principal = float(input("Investment amount ($): "))
                annual_return = float(input("Expected annual return (e.g., 0.10 for 10%): "))
                risk_score = float(input("Risk score (0-10): "))
                
                fund = MutualFundInvestment(name, principal, annual_return, risk_score)
                portfolio.add_investment(fund)
                print(f"\nMutual Fund '{name}' successfully added to portfolio!")
            
            # Register Cryptocurrency Investment
            elif choice == '4':
                if portfolio is None:
                    print("\nPlease initialize a portfolio first (Option 1)!")
                    continue
                
                name = input("\nEnter cryptocurrency name: ").strip()
                principal = float(input("Investment amount ($): "))
                annual_return = float(input("Expected annual return (e.g., 0.25 for 25%): "))
                risk_score = float(input("Risk score (0-10): "))
                
                crypto = CryptoInvestment(name, principal, annual_return, risk_score)
                portfolio.add_investment(crypto)
                print(f"\nCryptocurrency '{name}' successfully added to portfolio!")
            
            # Register Real Estate Investment
            elif choice == '5':
                if portfolio is None:
                    print("\nPlease initialize a portfolio first (Option 1)!")
                    continue
                
                name = input("\nEnter real estate property name: ").strip()
                principal = float(input("Investment amount ($): "))
                annual_return = float(input("Expected annual return (e.g., 0.06 for 6%): "))
                risk_score = float(input("Risk score (0-10): "))
                
                real_estate = RealEstateInvestment(name, principal, annual_return, risk_score)
                portfolio.add_investment(real_estate)
                print(f"\nReal Estate '{name}' successfully added to portfolio!")
            
            # Display Portfolio Summary
            elif choice == '6':
                if portfolio is None:
                    print("\nNo portfolio exists! Please create one first (Option 1).")
                    continue
                
                if len(portfolio.investments) == 0:
                    print("\nPortfolio is empty! Add some investments first.")
                    continue
                
                print("\n" + "-"*50)
                print("PORTFOLIO OVERVIEW")
                print("-"*50)
                print(f"Portfolio: {portfolio.portfolio_name}")
                print(f"Owner: {portfolio.owner.username} ({portfolio.owner.email})")
                print(f"Total Principal: ${portfolio.total_principal():.2f}")
                print(f"Projected Annual Return: ${portfolio.projected_annual_return():.2f}")
                print(f"Average Risk Score: {portfolio.risk_profile():.2f}")
                print(f"\nInvestments ({len(portfolio.investments)} total):")
                print("-" * 50)
                idx = 1
                for inv in portfolio.investments:
                    print(f"{idx}. {inv.name}")
                    print(f"   Principal: ${inv.principal:.2f}")
                    print(f"   Annual Return: {inv.annual_return*100:.2f}%")
                    print(f"   Risk Score: {inv.risk_score:.2f}")
                    print(f"   Projected Return: ${inv.calculate_return():.2f}")
                    idx += 1
                print("-"*50)
            
            # Identify Top Performing Investment
            elif choice == '7':
                if portfolio is None or len(portfolio.investments) == 0:
                    print("\nNo investments available for analysis!")
                    continue
                
                best = portfolio.best_investment()
                print("\n" + "-"*50)
                print("TOP PERFORMING INVESTMENT (Recursive Analysis)")
                print("-"*50)
                print(f"Investment: {best.name}")
                print(f"Annual Return Rate: {best.annual_return * 100:.2f}%")
                print(f"Principal Amount: ${best.principal:.2f}")
                print(f"Projected Return: ${best.calculate_return():.2f}")
                print(f"Total Investments Analyzed: {len(portfolio.investments)}")
                print("-"*50)
            
            # Perform Statistical Analysis
            elif choice == '8':
                if portfolio is None or len(portfolio.investments) == 0:
                    print("\nNo investments available for analysis!")
                    continue
                
                stats = portfolio.numpy_statistics()
                print("\n" + "-"*50)
                print("STATISTICAL ANALYSIS (NumPy)")
                print("-"*50)
                print(f"Average Annual Return: {stats['averageReturn']*100:.2f}%")
                print(f"Return Standard Deviation: {stats['riskStdDev']*100:.2f}%")
                print(f"Total Principal Invested: ${portfolio.total_principal():.2f}")
                print(f"Mean Risk Profile: {portfolio.risk_profile():.2f}")
                print(f"Total Projected Returns: ${portfolio.projected_annual_return():.2f}")
                print("-"*50)
            
            # Demonstrate Polymorphic Techniques
            elif choice == '9':
                if portfolio is None or len(portfolio.investments) < 2:
                    print("\nMinimum 2 investments required for polymorphism demonstration!")
                    continue
                
                print("\n" + "-"*50)
                print("POLYMORPHISM DEMONSTRATION")
                print("-"*50)
                
                inv1 = portfolio.investments[0]
                inv2 = portfolio.investments[1]
                
                print("\n1. Runtime Polymorphism (Method Overriding):")
                print(f"   {inv1.name}: ${PolymorphismShowcase.runtime_polymorphism(inv1):.2f}")
                print(f"   {inv2.name}: ${PolymorphismShowcase.runtime_polymorphism(inv2):.2f}")
                
                print("\n2. Ad-hoc Polymorphism (Function Overloading):")
                result = PolymorphismShowcase.ad_hoc_polymorphism(100, 50, 25, 10)
                print(f"   Combined value: ${result:.2f}")
                
                print("\n3. Operator Overloading:")
                composite = PolymorphismShowcase.operator_polymorphism(inv1, inv2)
                print(f"   Combined Investment: {composite.name}")
                print(f"   Composite Return: ${composite.calculate_return():.2f}")
                
                print("\n4. Duck Typing:")
                result = PolymorphismShowcase.duck_typing_polymorphism(inv1)
                print(f"   {result}")
                
                print("-"*50)
            
            # Persist Portfolio to File
            elif choice == '10':
                if portfolio is None:
                    print("\nNo portfolio available to save!")
                    continue
                
                PortfolioStorage.save(portfolio)
                print(f"\nPortfolio successfully saved to '{DATA_FILE}'!")
            
            # Generate Comprehensive Report
            elif choice == '11':
                if portfolio is None:
                    print("\nNo portfolio available for report generation!")
                    continue
                
                print("\n" + "-"*50)
                print("COMPREHENSIVE PORTFOLIO REPORT")
                print("-"*50)
                report = ReportGenerator.generate_summary(portfolio)
                print(report)
                print("-"*50)
            
            # Exit Program
            elif choice == '12':
                print("\n" + "-"*50)
                print("Thank you for using Smart Investment Portfolio Manager!")
                print("Your financial journey continues...")
                print("-"*50)
                break
            
            else:
                print("\nInvalid selection! Please choose a number between 1-12.")
        
        except PortfolioError as error:
            print(f"\nPortfolio Error: {error}")
        except ValueError as error:
            print(f"\nInput Error: Please enter valid numerical values.")
        except Exception as error:
            print(f"\nUnexpected Error: {error}")


if __name__ == "__main__":
    main()

