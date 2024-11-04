import smtplib
from email.mime.text import MIMEText


class AlertSystem:
    """
    System for handling metric alerts with different notification options.
    Supports console alerts, email notifications, and custom alert handlers.

    Parameters
    ----------
    email_config : dict, optional
        Configuration dictionary for email alerts, including SMTP server, port, username, and password.

    Examples
    --------
    >>> alert = AlertSystem(email_config={"server": "smtp.example.com", "port": 587, "username": "user", "password": "pass"})
    >>> alert.console_alert("accuracy", 0.75, 0.8, "less than")
    >>> alert.email_alert("recipient@example.com", "Accuracy Alert", "Accuracy has fallen below 80%.")
    """

    def __init__(self, email_config=None):
        self.email_config = email_config

    @staticmethod
    def console_alert(metric_name, value, threshold, condition):
        """
        Logs an alert message to the console.

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        value : float
            Current metric value.
        threshold : float
            Threshold for alert.
        condition : str
            Condition for triggering alert (e.g., "greater than", "less than").

        Example
        -------
        >>> AlertSystem.console_alert("accuracy", 0.75, 0.8, "less than")
        Alert: accuracy is less than threshold (0.8).
        """
        print(
            f"Alert: {metric_name} is {condition} threshold ({threshold}). Current value: {value}"
        )

    def email_alert(self, to_email, subject, message):
        """
        Sends an email alert.

        Parameters
        ----------
        to_email : str
            Recipient email address.
        subject : str
            Email subject.
        message : str
            Email message body.

        Example
        -------
        >>> alert.email_alert("recipient@example.com", "Accuracy Alert", "Accuracy has fallen below 80%.")
        """
        if not self.email_config:
            raise ValueError("Email configuration not provided.")

        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = self.email_config.get("username")
        msg["To"] = to_email

        with smtplib.SMTP(
            self.email_config.get("server"), self.email_config.get("port")
        ) as server:
            server.starttls()
            server.login(
                self.email_config.get("username"), self.email_config.get("password")
            )
            server.sendmail(
                self.email_config.get("username"), to_email, msg.as_string()
            )
