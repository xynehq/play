#!/usr/bin/env python3
import click

@click.group()
def main():
    """Xyne Play CLI"""
    pass

@main.command()
def hello():
    """Test command"""
    click.echo("hello")

if __name__ == '__main__':
    main()
